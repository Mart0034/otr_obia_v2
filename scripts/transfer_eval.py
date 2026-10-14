"""
Übertragungstest: bewertet das Chile-Modell an Orten außerhalb Chiles, die ein Mensch geprüft hat
===============================================================================================

Eingabe: reference/external_checks.csv (name,lat,lon,result,notes; result = dump oder none) und ein Lauf mit
segments_classified.gpkg der dazugehörigen Kacheln (ein Koordinatensystem). Für jeden Ort: höchster und mittlerer
dump_proba-Wert der Segmente im Umkreis, und wo dieser Wert in der Kachel und unter allen Kacheln des Laufs steht.

    python scripts/transfer_eval.py --checks reference/external_checks.csv --segments out_transfer/segments_classified.gpkg \\
        --radius-m 60 --out out_transfer/transfer_eval.csv
"""

import argparse
import sys

import geopandas as gpd
import numpy as np
import pandas as pd


def evaluate(checks, seg, radius_m=60.0, proba_col="dump_proba"):
    pts = gpd.GeoDataFrame(checks.copy(), geometry=gpd.points_from_xy(checks["lon"], checks["lat"]), crs=4326).to_crs(seg.crs)
    sindex = seg.sindex
    all_p = seg[proba_col].to_numpy()
    rows = []
    for _, r in pts.iterrows():
        hit = seg.iloc[list(sindex.query(r.geometry.buffer(radius_m), predicate="intersects"))]
        if not len(hit):
            rows.append({"name": r["name"], "result": r["result"], "segments": 0})
            continue
        tile = seg[seg["mine_id"] == hit["mine_id"].iloc[0]][proba_col].to_numpy()
        best = float(hit[proba_col].max())
        rows.append({"name": r["name"], "result": r["result"], "segments": int(len(hit)), "max_proba": round(best, 3),
                     "mean_proba": round(float(hit[proba_col].mean()), 3),
                     "pct_in_tile": round(float((tile < best).mean() * 100), 1),
                     "pct_in_run": round(float((all_p < best).mean() * 100), 1)})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checks", required=True)
    p.add_argument("--segments", required=True)
    p.add_argument("--radius-m", type=float, default=60.0)
    p.add_argument("--out")
    a = p.parse_args(argv)
    out = evaluate(pd.read_csv(a.checks), gpd.read_file(a.segments), a.radius_m)
    print(out.to_string(index=False))
    if "max_proba" in out:
        for res, g in out.groupby("result"):
            print(f"{res}: {len(g)} Orte, höchster Wert im Mittel {g['max_proba'].mean():.2f}, "
                  f"Perzentil in der Kachel im Mittel {g['pct_in_tile'].mean():.0f}")
    if a.out:
        out.to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
