"""
Trennen die vorhandenen Merkmale Reifenhalden von Erdhaufen-Fehlalarmen?
========================================================================

Nimmt die von dir geprüften Standorte (candidate_sites.gpkg mit `review`) und die Merkmale der Segmente aus dem
Kachel-Zwischenspeicher (--tile-cache-dir der Pipeline). Je Standort werden die Merkmale der Segmente darin gemittelt.
Verglichen werden

    Halden   review = dump oder partial
    Störer   review = clean, aber vom Modell hoch bewertet (mean_proba >= Schwelle: das sind die Fehlalarme, die in
             QGIS nerven - Erdhaufen, Terrassen, Straßen)

Ausgabe: je Merkmal die AUC (0.5 = trennt nichts, 1.0 = trennt perfekt; Richtung + Konfidenzintervall per Bootstrap),
dazu ein kleiner Test, wie gut ALLE Merkmale zusammen trennen (wiederholte Kreuzvalidierung). Aus den Zahlen folgt, ob
ein zweiter Filter auf den vorhandenen Merkmalen etwas bringt oder ob dafür andere Daten nötig sind.

Nutzung:
    python src/eval_site_features.py --reviewed rev_in/v4_top500.gpkg --tile-cache data/tile_cache \\
        --out-csv site_features_auc.csv
Hinweis: wenige Halden (ca. 20-30) -> Zahlen sind Hinweise, kein Beweis; die Intervalle zeigen die Unsicherheit.
"""

import argparse
import glob
import logging
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SKIP_COLS = {"segment_id", "label", "dump_proba", "dump_pred", "mine_id", "geometry", "oof_proba"}


def site_feature_table(sites, tile_cache_dir, id_col="site_id"):
    """Eine Zeile je Standort: Mittelwerte aller numerischen Segment-Merkmale der Segmente, deren Mittelpunkt im
    Standort liegt (Segmente und Standort in derselben Kachel = mine_id)."""
    rows = []
    by_tile = {m: g for m, g in sites.groupby("mine_id")}
    for mine, grp in by_tile.items():
        files = sorted(glob.glob(os.path.join(tile_cache_dir, f"{mine}.*.pkl")))
        if not files:
            continue
        feats, polys = pd.read_pickle(files[-1])
        polys = gpd.GeoDataFrame(polys)
        if polys.crs is not None and sites.crs is not None and polys.crs != sites.crs:
            polys = polys.to_crs(sites.crs)
        cent = gpd.GeoDataFrame({"segment_id": polys["segment_id"].values}, geometry=polys.geometry.centroid.values,
                                crs=polys.crs)
        num = [c for c in feats.columns if c not in SKIP_COLS and pd.api.types.is_numeric_dtype(feats[c])]
        feats = feats[["segment_id"] + num]
        hit = gpd.sjoin(cent, grp[[id_col, "geometry"]], predicate="within", how="inner")
        merged = hit[["segment_id", id_col]].merge(feats, on="segment_id", how="left")
        agg = merged.groupby(id_col)[num].mean()
        agg["n_seg_features"] = merged.groupby(id_col).size()
        rows.append(agg)
    return pd.concat(rows) if rows else pd.DataFrame()


def auc_with_ci(pos, neg, n_boot=300, seed=0):
    """AUC (Wahrscheinlichkeit, dass ein zufälliger Treffer einen höheren Wert hat als ein Störer) mit
    Bootstrap-Intervall; NaN-Werte werden ignoriert."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan, np.nan, np.nan

    def auc(p, n):
        ranks = pd.Series(np.concatenate([p, n])).rank().values
        return (ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))

    rng = np.random.default_rng(seed)
    boots = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    return auc(pos, neg), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def feature_auc_table(table, is_pos):
    rows = []
    for c in table.columns:
        a, lo, hi = auc_with_ci(table.loc[is_pos, c], table.loc[~is_pos, c])
        if np.isnan(a):
            continue
        rows.append({"merkmal": c, "auc": round(a, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                     "richtung": "Halde höher" if a >= 0.5 else "Halde niedriger",
                     "stärke": round(abs(a - 0.5) * 2, 3), "sicher": bool(lo > 0.5 or hi < 0.5),
                     "mittel_halde": round(float(table.loc[is_pos, c].mean()), 3),
                     "mittel_stoerer": round(float(table.loc[~is_pos, c].mean()), 3)})
    return pd.DataFrame(rows).sort_values("stärke", ascending=False).reset_index(drop=True)


def combined_cv_auc(table, is_pos, repeats=10, seed=0):
    """AUC eines Random Forest auf allen Merkmalen zusammen, wiederholte stratifizierte Kreuzvalidierung."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    x = table.fillna(table.median()).fillna(0).values
    y = np.asarray(is_pos, int)
    k = int(min(5, y.sum(), (1 - y).sum()))
    if k < 2:
        return np.nan, np.nan
    scores = []
    for r in range(repeats):
        pred = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed + r).split(x, y):
            clf = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=seed + r, n_jobs=-1)
            pred[te] = clf.fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
        scores.append(roc_auc_score(y, pred))
    return float(np.mean(scores)), float(np.std(scores))


def main(argv=None):
    p = argparse.ArgumentParser(description="Trennen die Segment-Merkmale Halden von Erdhaufen-Fehlalarmen?")
    p.add_argument("--reviewed", required=True, help="candidate_sites.gpkg mit Spalte review (und mean_proba).")
    p.add_argument("--tile-cache", required=True, help="Kachel-Zwischenspeicher der Pipeline (--tile-cache-dir).")
    p.add_argument("--confuser-min-proba", type=float, default=None,
                   help="Störer = geprüft clean UND mean_proba >= dieser Wert (Standard: unteres Quartil der Halden).")
    p.add_argument("--top", type=int, default=25, help="Wie viele Merkmale ausgeben.")
    p.add_argument("--out-csv", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sites = gpd.read_file(args.reviewed, layer="candidate_sites")
    sites["review"] = sites["review"].fillna("").astype(str).str.strip().str.lower()
    pos_mask = sites["review"].isin(["dump", "partial"])
    thr = args.confuser_min_proba if args.confuser_min_proba is not None else float(
        sites.loc[pos_mask, "mean_proba"].quantile(0.25))
    conf_mask = (sites["review"] == "clean") & (sites["mean_proba"] >= thr)
    use = sites[pos_mask | conf_mask].copy()
    logger.info("Halden (dump + partial): %d, Störer (clean, mean_proba >= %.2f): %d", int(pos_mask.sum()), thr, int(conf_mask.sum()))
    table = site_feature_table(use, args.tile_cache)
    use = use[use["site_id"].isin(table.index)].set_index("site_id")
    table = table.loc[use.index]
    is_pos = use["review"].isin(["dump", "partial"]).values
    logger.info("Standorte mit Merkmalen: %d (Halden %d, Störer %d)", len(table), int(is_pos.sum()), int((~is_pos).sum()))
    res = feature_auc_table(table.drop(columns=["n_seg_features"], errors="ignore"), pd.Series(is_pos, index=table.index))
    cv, sd = combined_cv_auc(table.drop(columns=["n_seg_features"], errors="ignore"), is_pos)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 200):
        print(res.head(args.top).to_string(index=False))
    print(f"\nAlle Merkmale zusammen (Random Forest, wiederholte Kreuzvalidierung): AUC {cv:.3f} +- {sd:.3f}")
    print(f"Zum Vergleich - Modell-Bewertung selbst (mean_proba) trennt Halden von Störern bei diesen Standorten: "
          f"AUC {auc_with_ci(use.loc[is_pos, 'mean_proba'], use.loc[~is_pos, 'mean_proba'])[0]:.3f}")
    if args.out_csv:
        res.to_csv(args.out_csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
