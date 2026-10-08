"""
In QGIS geprüfte Kandidaten-Standorte in Trainingsdaten umwandeln
=================================================================

Eingabe ist die candidate_sites.gpkg, in der in QGIS die Spalte `review`
ausgefüllt wurde:

    dump    -> echte Reifenhalde      (neues Positivbeispiel)
    clean   -> keine Halde            (bestätigtes Negativbeispiel / Fehlalarm)
    partial -> enthält eine Halde, aber der Umriss ist viel größer als die Halde
               (nicht als Positiv/Negativ verwenden; wird beim Training ignoriert -
               besser: den Umriss in QGIS auf die Halde zurechtziehen und dann dump)
    unsure  -> unklar                 (wird beim Training ignoriert)
    (leer)  -> noch nicht geprüft

Zusätzlich kann in QGIS in der Schicht `drawn_polygons` (gleiche Datei) von Hand
eingezeichnet werden, was genau Halde (kind = dump) bzw. eindeutig sauber
(kind = clean) ist. Gezeichnete Polygone haben Vorrang: ein Standort, den sie
berühren, liefert nicht mehr seinen ganzen Umriss als Label, sondern nur die
gezeichneten Umrisse. Ohne mine_id wird sie vom nächsten Standort übernommen.

Ausgabe in --out-dir:
    labels_reviewed_dump.gpkg      Halden-Polygone (mine_id = Kachel/Mine) - wie
                                   --extra-labels-path bzw. dump_labels.gpkg
    negatives_reviewed_clean.gpkg  bestätigte Fehlalarme (mine_id, site_id)
    ignore_reviewed.geojson        Umrisse der unklaren/teilweisen Standorte (--ignore-points-path;
                                   in dieser Zone werden Negativ-Segmente nicht mittrainiert)
    review_summary.csv             geprüfte Standorte mit Rang und Urteil

und eine Trefferquote nach Rang ("von den ersten k geprüften Standorten waren
x % echte Halden") - das ist die ehrliche Antwort auf "wie viele der obersten
Funde stimmen?".

Nutzung:
    python src/reviewed_to_labels.py --reviewed candidate_sites_reviewed.gpkg --out-dir reviewed_out
"""

import argparse
import os
import sys

import geopandas as gpd
import pandas as pd
import pyogrio

VALID = ("dump", "clean", "partial", "unsure")


def _clean_drawn(drawn, sites, max_dist_m=500.0):
    """Prüft die gezeichneten Polygone und ergänzt fehlende mine_id vom nächsten Standort."""
    drawn = drawn[drawn.geometry.notna() & ~drawn.geometry.is_empty].copy()
    if drawn.empty:
        return drawn
    drawn["kind"] = drawn["kind"].fillna("").astype(str).str.strip().str.lower()
    bad = sorted(set(drawn["kind"]) - {"dump", "clean"})
    if bad:
        raise ValueError(f"Unbekannte Werte in drawn_polygons.kind: {bad} (erlaubt: dump, clean)")
    drawn["mine_id"] = drawn["mine_id"].fillna("").astype(str).str.strip()
    missing = drawn["mine_id"] == ""
    if missing.any():
        near = gpd.sjoin_nearest(drawn[missing][["geometry"]], sites[["mine_id", "geometry"]],
                                 how="left", max_distance=max_dist_m)
        near = near[~near.index.duplicated(keep="first")]
        if near["mine_id"].isna().any():
            raise ValueError(
                f"{int(near['mine_id'].isna().sum())} gezeichnete Polygone liegen weiter als "
                f"{max_dist_m:.0f} m von jedem Standort entfernt und haben keine mine_id - bitte "
                "mine_id eintragen (z.B. mine_019 bzw. der Kachelname).")
        drawn.loc[missing, "mine_id"] = near["mine_id"]
    return drawn


def apply_drawn(sites, drawn):
    """Gezeichnete Polygone haben Vorrang vor dem Urteil über den ganzen Standort.
    Gibt (sites, drawn, skip) zurück: skip = Standorte, die gezeichnete Polygone
    berühren (ihr Umriss wird nicht als Label verwendet). Ein Standort ohne eigenes
    Urteil bekommt: dump (gezeichnete Halde füllt >= die Hälfte), partial (weniger),
    clean (nur gezeichnete saubere Flächen)."""
    sites = sites.copy()
    skip = pd.Series(False, index=sites.index)
    drawn = _clean_drawn(drawn, sites)
    if drawn.empty:
        return sites, drawn, skip
    for idx, site in sites.iterrows():
        hit = drawn[drawn.geometry.intersects(site.geometry)]
        if hit.empty:
            continue
        skip[idx] = True
        if str(site["review"]).strip().lower() not in ("", "none", "nan"):
            continue
        dump_area = hit[hit["kind"] == "dump"].geometry.intersection(site.geometry).area.sum()
        if dump_area > 0:
            sites.loc[idx, "review"] = "dump" if dump_area >= 0.5 * site.geometry.area else "partial"
        else:
            sites.loc[idx, "review"] = "clean"
    return sites, drawn, skip


def split_reviews(sites, close_gaps_m=5.0, drawn=None):
    """Teilt die Standorte nach dem Urteil in (dump, clean, unsure, summary).
    dump/clean: GeoDataFrame(mine_id, site_id, geometry); unsure: Umrisse der
    unklaren und teilweisen Standorte (Ignorier-Zone)."""
    sites = sites.copy()
    sites["review"] = sites["review"].fillna("").astype(str).str.strip().str.lower()
    skip = pd.Series(False, index=sites.index)
    drawn_dump = drawn_clean = None
    if drawn is not None and len(drawn):
        sites, drawn, skip = apply_drawn(sites, drawn.to_crs(sites.crs) if drawn.crs else drawn)
        sites["review"] = sites["review"].astype(str).str.strip().str.lower()
        cols_d = ["mine_id", "geometry"]
        drawn_dump = drawn[drawn["kind"] == "dump"][cols_d].assign(site_id="drawn")
        drawn_clean = drawn[drawn["kind"] == "clean"][cols_d].assign(site_id="drawn")
    unknown = sorted(set(sites["review"]) - set(VALID) - {""})
    if unknown:
        raise ValueError(f"Unbekannte Werte in 'review': {unknown} (erlaubt: {', '.join(VALID)} oder leer)")
    cols = ["mine_id", "site_id"]
    # Lücken zwischen den markierten Segmenten schließen, damit ein zusammenhängendes
    # Halden-Polygon entsteht
    shaped = sites.assign(geometry=sites.geometry.buffer(close_gaps_m).buffer(-close_gaps_m))
    dump = shaped[(sites["review"] == "dump") & ~skip][cols + ["geometry"]]
    clean = shaped[(sites["review"] == "clean") & ~skip][cols + ["geometry"]]
    if drawn_dump is not None:
        dump = pd.concat([dump, drawn_dump[cols + ["geometry"]]], ignore_index=True)
        clean = pd.concat([clean, drawn_clean[cols + ["geometry"]]], ignore_index=True)
        dump, clean = gpd.GeoDataFrame(dump, crs=sites.crs), gpd.GeoDataFrame(clean, crs=sites.crs)
    # unklare und teilweise Standorte: als Ganzes aus dem Negativ-Training nehmen
    unsure = sites[sites["review"].isin(["unsure", "partial"])][cols + ["geometry"]].copy()
    summary = sites[sites["review"] != ""].sort_values("rank")[
        [c for c in ("rank", "site_id", "mine_id", "size_class", "area_m2", "mean_proba", "review", "note")
         if c in sites.columns]
    ]
    return dump, clean, unsure, summary


def precision_by_rank(summary, steps=(5, 10, 25, 50, 100, 250, 500)):
    """Anteil echter Halden unter den geprüften Standorten, die bis Rang k geprüft
    wurden. Es zählen dump, clean und partial (unsichere nicht). Streng: nur dump
    ist ein Treffer; großzügig: partial zählt auch (eine Halde ist enthalten)."""
    decided = summary[summary["review"].isin(["dump", "clean", "partial"])]
    rows = []
    for k in steps:
        upto = decided[decided["rank"] <= k]
        if len(upto):
            rows.append({"bis_rang": k, "geprueft": len(upto),
                         "davon_dump": int((upto["review"] == "dump").sum()),
                         "davon_partial": int((upto["review"] == "partial").sum()),
                         "trefferquote": float((upto["review"] == "dump").mean()),
                         "trefferquote_mit_partial": float(upto["review"].isin(["dump", "partial"]).mean())})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Geprüfte Kandidaten-Standorte in Labels umwandeln.")
    p.add_argument("--reviewed", required=True, help="candidate_sites.gpkg mit ausgefüllter Spalte 'review'.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    sites = gpd.read_file(args.reviewed, layer="candidate_sites") if args.reviewed.endswith(".gpkg") \
        else gpd.read_file(args.reviewed)
    drawn = None
    if args.reviewed.endswith(".gpkg") and "drawn_polygons" in set(pyogrio.list_layers(args.reviewed)[:, 0]):
        drawn = gpd.read_file(args.reviewed, layer="drawn_polygons")
        print(f"Gezeichnete Polygone: {len(drawn)}")
    dump, clean, unsure, summary = split_reviews(sites, drawn=drawn)
    os.makedirs(args.out_dir, exist_ok=True)
    if len(dump):
        dump.to_file(os.path.join(args.out_dir, "labels_reviewed_dump.gpkg"), driver="GPKG")
    if len(clean):
        clean.to_file(os.path.join(args.out_dir, "negatives_reviewed_clean.gpkg"), driver="GPKG")
    if len(unsure):
        unsure.to_crs(4326).to_file(os.path.join(args.out_dir, "ignore_reviewed.geojson"), driver="GeoJSON")
    summary.to_csv(os.path.join(args.out_dir, "review_summary.csv"), index=False)
    print(f"Geprüft: {len(summary)} von {len(sites)} Standorten -> dump {len(dump)}, "
          f"clean {len(clean)}, unsicher/teilweise {len(unsure)}")
    table = precision_by_rank(summary)
    if len(table):
        print("\nTrefferquote nach Rang (dump/clean/partial):")
        pct = lambda col: (100 * table[col]).round(0).astype(int).astype(str) + " %"
        print(table.assign(trefferquote=pct("trefferquote"),
                           trefferquote_mit_partial=pct("trefferquote_mit_partial")).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
