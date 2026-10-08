"""
Wie viele bestätigte Fehlalarme markiert ein Lauf noch?
=======================================================

Ergänzung zu eval_segment_size.py: dort zählt, wie viele BEKANNTE Halden gefunden
werden. Hier zählt, wie viel von den in QGIS als `clean` bestätigten Flächen
(Fehlalarme) noch in der obersten X % Fläche je Mine landet - weniger ist besser.
Beide Zahlen zusammen zeigen, ob eine Änderung (z.B. harte Negative im Training)
Fehlalarme senkt, ohne echte Halden zu verlieren.

Grundlage sind Out-of-Fold-Vorhersagen (jede Mine von einem Modell bewertet, das sie
nie sah) - die Fehlalarme einer Mine haben das Modell, das sie bewertet, also nicht
mittrainiert.

Nutzung:
    python src/eval_false_alarms.py --negatives data/reviewed/out/negatives_all_clean.gpkg \\
        --run base=out_px12_hn1 --run hn10=out_px12_hn10
"""

import argparse
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from eval_segment_size import load_run


def false_alarm_share(oof, polygons, negatives, qs=(1, 2, 5, 10)):
    """Für jedes q: (Anteil der Fehlalarm-FLÄCHE, Anteil der Fehlalarm-STANDORTE), die von
    den am höchsten bewerteten Segmenten (je Mine bis q % der Mine) berührt werden;
    außerdem die mittlere Bewertung (dump_proba) auf den Fehlalarm-Flächen."""
    seg = polygons.merge(oof[["mine_id", "segment_id", "dump_proba"]], on=["mine_id", "segment_id"])
    seg = seg[seg["dump_proba"].notna()].assign(area=lambda t: t.geometry.area)
    seg["mine_id"] = seg["mine_id"].astype(str)
    negatives = negatives.to_crs(seg.crs).assign(mine_id=lambda t: t["mine_id"].astype(str).str.strip())
    out = {}
    for q in qs:
        a_hit = a_total = 0.0
        n_hit = n_total = 0
        for mine, neg in negatives.groupby("mine_id"):
            s = seg[seg["mine_id"] == mine]
            if s.empty:
                continue
            s = s.sort_values("dump_proba", ascending=False)
            keep = s["area"].cumsum() - s["area"] < s["area"].sum() * q / 100.0
            flagged = s[keep.values].geometry.union_all()
            a_hit += float(neg.geometry.intersection(flagged).area.sum())
            a_total += float(neg.geometry.area.sum())
            n_hit += int(neg.geometry.intersects(flagged).sum())
            n_total += len(neg)
        out[q] = (a_hit / max(a_total, 1e-9), n_hit / max(n_total, 1))
    # mittlere Bewertung auf den Fehlalarm-Flächen
    scores = []
    for mine, neg in negatives.groupby("mine_id"):
        s = seg[seg["mine_id"] == mine]
        if s.empty:
            continue
        inside = s[s.geometry.centroid.within(neg.geometry.union_all())]
        scores.append(inside["dump_proba"])
    mean_score = float(pd.concat(scores).mean()) if scores else float("nan")
    return out, mean_score


def main(argv=None):
    p = argparse.ArgumentParser(description="Anteil bestätigter Fehlalarme in der obersten X%% Fläche.")
    p.add_argument("--negatives", required=True, help="Bestätigte Fehlalarme (GeoPackage, mine_id).")
    p.add_argument("--run", action="append", required=True, metavar="NAME=ORDNER")
    args = p.parse_args(argv)
    neg = gpd.read_file(args.negatives)
    qs = (1, 2, 5, 10)
    print("Anteil der bestätigten Fehlalarm-Fläche / -Standorte in der obersten X% Fläche je Mine (weniger = besser)")
    print("%-14s" % "" + "".join(f"{'top %d%%' % q:>16}" for q in qs) + f"{'mean proba':>12}")
    for item in args.run:
        name, folder = item.split("=", 1)
        oof, poly = load_run(folder)
        res, mean_score = false_alarm_share(oof, poly, neg, qs)
        print("%-14s" % name + "".join(f"{100 * res[q][0]:>8.0f}% /{100 * res[q][1]:>4.0f}%" for q in qs)
              + f"{mean_score:>12.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
