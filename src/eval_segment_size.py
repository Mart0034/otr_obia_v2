"""
Segmentgrößen vergleichen: wie viele bekannte Halden fängt ein Lauf?
=====================================================================

Verschiedene Segmentgrößen (--target-segment-px) liefern verschieden viele
Segmente und verschieden definierte Positive, deshalb sind Segment-Zahlen
(F1, "positive Segmente") zwischen Läufen nicht vergleichbar. Dieser Vergleich
ist größenunabhängig: pro Mine werden die am höchsten bewerteten Segmente so
lange markiert, bis q% der gültigen Fläche der Mine erreicht sind; gezählt wird,
wie viele der bekannten Halden-Polygone (data/dump_labels.gpkg) von markierter
Fläche berührt werden (und wie viel Halden-FLÄCHE).

Eingabe sind Out-of-Fold-Vorhersagen (oof_predictions.pkl, vom Trainingslauf
geschrieben) - jede Mine wurde von einem Modell bewertet, das sie nie sah.

Nutzung:
    python src/eval_segment_size.py --labels data/dump_labels.gpkg \\
        --run baseline=output_a --run px60=output_b
"""

import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd


def area_recall(oof, polygons, labels, qs=(1, 2, 5, 10)):
    """oof: mine_id, segment_id, dump_proba (NaN-Zeilen werden ignoriert).
    polygons: mine_id, segment_id, geometry (Segment-Polygone).
    labels: Halden-Polygone mit Spalte mine_id.
    Gibt pro q: (Anteil gefangener Halden-Polygone, Anteil gefangener Halden-Fläche)."""
    seg = polygons.merge(oof[["mine_id", "segment_id", "dump_proba"]], on=["mine_id", "segment_id"])
    seg = seg[seg["dump_proba"].notna()]
    seg = seg.assign(area=seg.geometry.area)
    labels = labels.to_crs(seg.crs) if seg.crs is not None else labels
    labels = labels.assign(mine_id=labels["mine_id"].astype(str).str.strip())
    seg["mine_id"] = seg["mine_id"].astype(str)
    out = {}
    for q in qs:
        n_caught = n_total = 0
        a_caught = a_total = 0.0
        for mine, dumps in labels.groupby("mine_id"):
            s = seg[seg["mine_id"] == mine]
            if s.empty:
                continue
            s = s.sort_values("dump_proba", ascending=False)
            keep = s["area"].cumsum() - s["area"] < s["area"].sum() * q / 100.0
            flagged = s[keep.values].geometry.union_all()
            hit = dumps.geometry.intersects(flagged)
            n_caught += int(hit.sum())
            n_total += len(dumps)
            a_total += float(dumps.geometry.area.sum())
            a_caught += float(dumps.geometry.intersection(flagged).area.sum())
        out[q] = (n_caught / max(n_total, 1), a_caught / max(a_total, 1e-9))
    return out


def load_run(folder):
    """oof_predictions.pkl und die Segment-Polygone eines Laufs. Bei Läufen mit
    zusätzlichen Kacheln (--labeled/--extra-imagery-dir) kommen deren Polygone
    aus polygons_extra.gpkg dazu."""
    oof = pd.read_pickle(f"{folder}/oof_predictions.pkl")
    poly = gpd.read_file(f"{folder}/polygons_cache.gpkg")
    extra = f"{folder}/polygons_extra.gpkg"
    if os.path.exists(extra):
        poly = pd.concat([poly, gpd.read_file(extra)], ignore_index=True)
    return oof, poly


def main(argv=None):
    p = argparse.ArgumentParser(description="Größenunabhängiger Vergleich mehrerer Läufe.")
    p.add_argument("--labels", required=True, help="Halden-Polygone (GeoPackage).")
    p.add_argument("--run", action="append", required=True, metavar="NAME=ORDNER",
                   help="Ausgabeordner eines Laufs (mit oof_predictions.pkl und polygons_cache.gpkg).")
    args = p.parse_args(argv)
    labels = gpd.read_file(args.labels)
    qs = (1, 2, 5, 10)
    print("Anteil bekannter Halden-Polygone / Halden-Fläche, die in der obersten X% Fläche je Mine liegen")
    print("%-14s" % "" + "".join(f"{'top %d%%' % q:>16}" for q in qs))
    for item in args.run:
        name, folder = item.split("=", 1)
        oof, poly = load_run(folder)
        res = area_recall(oof, poly, labels, qs)
        print("%-14s" % name + "".join(f"{100 * res[q][0]:>8.0f}% /{100 * res[q][1]:>4.0f}%" for q in qs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
