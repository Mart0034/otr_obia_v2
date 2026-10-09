"""
Zwei-Stufen-Test: wie viel Fläche lässt sich mit billigen Daten verwerfen?
==========================================================================

Stufe 1 nutzt nur billige Merkmale (S1-Radar, DEM, Form, optional Mehrjahres-
Helligkeit), Stufe 2 (volle optische Bilder, Textur) liefe nur auf dem Rest.
Gemessen wird mit Out-of-Fold-Vorhersagen (GroupKFold nach Mine): bei welchem
Anteil behaltener Fläche bleiben 99/98/95/90 % der bekannten Halden-Polygone
erhalten? Die Halde zählt als behalten, wenn mindestens EIN sie berührendes
Segment die Schwelle erreicht.

    python src/eval_stage1_filter.py --cache-dir out_px12_my_bright \\
        --labels data/dump_labels.gpkg --add-labels data/reviewed/out/labels_reviewed_dump_all.gpkg
"""
import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otr_obia_pipeline import apply_extra_labels  # noqa: E402

NON_FEATURE = ("segment_id", "mine_id", "label", "overlap_area", "overlap_ratio")
CHEAP_PREFIXES = ("s1_", "dem_", "shape_", "osm_")


def feature_sets(columns):
    cols = [c for c in columns if c not in NON_FEATURE]
    pick = lambda prefixes: [c for c in cols if c.startswith(prefixes)]
    base = pick(CHEAP_PREFIXES)
    my = pick(("s2my_",))
    return {
        "billig: S1+DEM+Form": base,
        "billig + Mehrjahres-Helligkeit": base + my,
        "alles (Referenz)": cols,
    }


def oof_proba(df, cols, n_trees, n_jobs, seed=0):
    X = df[cols].fillna(0.0).values
    y = df["label"].values
    groups = df["mine_id"].values
    out = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        clf = BalancedRandomForestClassifier(
            n_estimators=n_trees, n_jobs=n_jobs, random_state=seed,
            sampling_strategy="auto", replacement=True)
        clf.fit(X[tr], y[tr])
        out[va] = clf.predict_proba(X[va])[:, 1]
    return out


def kept_curve(seg, labels, targets=(0.99, 0.98, 0.95, 0.90)):
    """seg: GeoDataFrame mit mine_id, proba, area. labels: Halden-Polygone."""
    labels = labels.to_crs(seg.crs).reset_index(drop=True)
    labels["lab_id"] = labels.index
    labels["mine_id"] = labels["mine_id"].astype(str).str.strip()
    j = gpd.sjoin(labels[["lab_id", "mine_id", "geometry"]],
                  seg[["mine_id", "proba", "geometry"]], predicate="intersects")
    j = j[j["mine_id_left"] == j["mine_id_right"]]
    lab_max = j.groupby("lab_id")["proba"].max()
    lab_max = lab_max.reindex(labels["lab_id"]).fillna(-1.0)  # nie berührt -> verloren
    covered = (lab_max >= 0).sum()
    order = np.sort(lab_max[lab_max >= 0].values)[::-1]
    p, a = seg["proba"].values, seg["area"].values
    total = a.sum()
    rows = []
    for t in targets:
        need = int(np.ceil(t * len(labels)))
        if need > len(order):
            rows.append((t, np.nan, np.nan)); continue
        thr = order[need - 1]
        rows.append((t, thr, a[p >= thr].sum() / total))
    return rows, covered, len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--labels", required=True, help="Halden-Polygone für die Bewertung")
    ap.add_argument("--add-labels", default=None, help="Zusatz-Labels fürs Training")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--min-overlap", type=float, default=0.1)
    a = ap.parse_args()

    df = pd.read_pickle(os.path.join(a.cache_dir, "feature_cache.pkl"))
    polys = gpd.read_file(os.path.join(a.cache_dir, "polygons_cache.gpkg"))
    if a.add_labels:
        df = apply_extra_labels(df, polys, gpd.read_file(a.add_labels), a.min_overlap)
    labels = gpd.read_file(a.labels)
    print(f"{len(df)} Segmente, {int(df['label'].sum())} positiv, {df['mine_id'].nunique()} Minen, "
          f"{len(labels)} bewertete Halden-Polygone", flush=True)

    seg = polys[["mine_id", "segment_id", "geometry"]].merge(
        df[["mine_id", "segment_id"]].assign(_i=np.arange(len(df))), on=["mine_id", "segment_id"])
    seg["area"] = seg.geometry.area
    idx = seg["_i"].values

    for name, cols in feature_sets(df.columns).items():
        proba = oof_proba(df, cols, a.trees, a.n_jobs)
        seg["proba"] = proba[idx]
        ok = seg[seg["proba"].notna()]
        rows, covered, n = kept_curve(ok, labels)
        print(f"\n== {name} ({len(cols)} Merkmale) | Halden von Segmenten berührt: {covered}/{n}", flush=True)
        print("  Ziel: Halden behalten -> Anteil der Fläche, die für Stufe 2 übrig bleibt")
        for t, thr, frac in rows:
            print(f"  {t:4.0%} Halden behalten -> {frac:6.1%} der Fläche bleibt "
                  f"(Schwelle {thr:.3f}, {1 - frac:5.1%} verworfen)" if frac == frac else f"  {t:4.0%}: nicht erreichbar")


if __name__ == "__main__":
    main()
