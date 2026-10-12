"""
Wie nah liegen die Markierungen des Modells an den echten Halden?
=================================================================

Misst für jede bekannte Halde den Abstand zur nächsten Markierung (je Mine die obersten
X % der Fläche, wie in den Standortlisten). Daraus folgt, wie groß der Umkreis sein muss, den
man sich in QGIS pro Standort ansehen will, und wie viele Halden "knapp daneben" liegen:

    Treffer      Markierung überdeckt die Halde (Abstand 0 m)
    knapp        Abstand 1 .. 100/250/500/1000 m
    nicht        weiter weg, oder in der Mine wurde nichts markiert

Das Skript braucht keine Bilder und keine API - nur die Segmentdatei eines Laufs (mit
dump_proba, ideal Out-of-Fold) und die bekannten Halden (--labels, mit mine_id).

Nutzung:
    python src/eval_near_miss.py --segments out_px12_qv2/segments_classified.gpkg \\
        --labels data/reviewed/out/labels_all3_for_eval.gpkg --area-pct 2 5 \\
        --out-csv near_miss_v2.csv

Standortlisten (--sites) nur verwenden, wenn sie OHNE --exclude-known gebaut wurden: sonst fehlen
gerade die Standorte nahe bekannter Halden, und die Abstände fallen zu groß aus.
"""

import argparse
import logging
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

THRESHOLDS = (0, 100, 250, 500, 1000)
SIZE_BINS = ((0, 500, "<500 m2"), (500, 2000, "500-2000 m2"), (2000, float("inf"), ">2000 m2"))


def nearest_distances(labels, flagged, same_mine=True):
    """Abstand (m) jeder Halde zur nächsten markierten Fläche; 0 = überdeckt. NaN, wenn es dafür
    keine Markierung gibt (bei same_mine: in der Mine der Halde wurde nichts markiert).
    labels/flagged: GeoDataFrames im selben projizierten CRS; bei same_mine mit Spalte mine_id."""
    labels = labels.reset_index(drop=True)
    dist = pd.Series(np.nan, index=labels.index)
    if flagged is None or flagged.empty:
        return dist
    groups = [(None, labels, flagged)]
    if same_mine:
        groups = [(m, labels[labels["mine_id"] == m], flagged[flagged["mine_id"] == m])
                  for m in labels["mine_id"].unique()]
    for _, lab, fl in groups:
        if lab.empty or fl.empty:
            continue
        near = gpd.sjoin_nearest(lab[["geometry"]], fl[["geometry"]], how="left", distance_col="_d")
        dist.loc[lab.index] = near.groupby(level=0)["_d"].min().reindex(lab.index).values
    return dist


def summarize(labels, dist, thresholds=THRESHOLDS):
    """Anteil der Halden mit Abstand <= t (t = 0: Treffer) - insgesamt und je Größenklasse der Halde."""
    area = labels.reset_index(drop=True).geometry.area
    rows = []
    for name, mask in [("alle", np.ones(len(dist), dtype=bool))] + \
            [(label, ((area >= lo) & (area < hi)).values) for lo, hi, label in SIZE_BINS]:
        d = dist[mask]
        if not len(d):
            continue
        row = {"gruppe": name, "n": int(len(d)), "ohne_markierung": int(d.isna().sum()),
               "median_m": float(d.median()) if d.notna().any() else np.nan}
        for t in thresholds:
            row[f"<= {t} m"] = round(float((d <= t).sum()) / len(d), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Abstand bekannter Halden zu den Markierungen des Modells.")
    p.add_argument("--labels", required=True, help="Bekannte Halden (Polygone, mine_id, projiziert oder wird umgerechnet).")
    p.add_argument("--segments", help="segments_classified.gpkg eines Laufs (je Mine die obersten area-pct %).")
    p.add_argument("--area-pct", type=float, nargs="+", default=[2.0, 5.0],
                   help="Anteil der Fläche je Mine, der als markiert gilt (mehrere Werte = mehrere Zeilen).")
    p.add_argument("--sites", nargs="*", default=[], help="Standortlisten (.gpkg), OHNE --exclude-known gebaut.")
    p.add_argument("--out-csv", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    labels = gpd.read_file(args.labels)
    tables = []
    if args.segments:
        from candidate_sites import load_top_segments, select_seeds
        for pct in args.area_pct:
            seg = load_top_segments(args.segments, pct, None, fraction_margin=1.0)
            labels = labels.to_crs(seg.crs)
            flagged = select_seeds(seg, None, pct)
            logger.info("Top %.1f%% Fläche je Mine: %d Segmente markiert.", pct, len(flagged))
            tab = summarize(labels, nearest_distances(labels, flagged, same_mine=True))
            tab.insert(0, "markierung", f"top {pct:g}% Fläche je Mine")
            tables.append(tab)
    for path in args.sites:
        sites = gpd.read_file(path).to_crs(labels.crs)
        tab = summarize(labels, nearest_distances(labels, sites, same_mine=False))
        tab.insert(0, "markierung", path)
        tables.append(tab)
    if not tables:
        p.error("Entweder --segments oder --sites angeben.")
    out = pd.concat(tables, ignore_index=True)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))
    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
