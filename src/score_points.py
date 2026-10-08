"""
Bewertung an bekannten Koordinaten ablesen
==========================================

Für eine Liste von Punkten (z.B. bestätigte Halden aus Google Maps) zeigt das
Skript, wie das Modell sie bewertet hat: Wert des Segments am Punkt, Maximum im
Umkreis und - am anschaulichsten - in welchem Anteil der Kachelfläche der Punkt
liegt, wenn man die Segmente von oben nach unten markiert ("top 3 %" = nur 3 %
der Kachel hatten eine gleich hohe oder höhere Bewertung).

Nutzung:
    python src/score_points.py --segments out/segments_classified.gpkg \\
        --points data/new/new_points.geojson --out out/point_scores.csv
"""

import argparse
import sys

import geopandas as gpd
import numpy as np
import pandas as pd


def score_points(seg, points, radii=(100, 250)):
    """seg: Segmente mit mine_id, dump_proba, geometry (projiziertes CRS).
    points: Punkte (beliebiges CRS). Eine Zeile je Punkt."""
    pts = points.to_crs(seg.crs).reset_index(drop=True)
    seg = seg[seg["dump_proba"].notna()].copy()
    seg["_area"] = seg.geometry.area
    rows = []
    tree = seg.sindex
    for i, p in enumerate(pts.geometry):
        # intersects statt contains: ein Punkt genau auf einer Segmentgrenze (z.B. der
        # Mittelpunkt einer Ein-Punkt-Kachel liegt auf einer Pixelecke) gehört zu
        # keinem Segment "innen". Bei mehreren berührten Segmenten zählt das beste.
        hit = seg.iloc[tree.query(p, predicate="intersects")]
        hit = hit.sort_values("dump_proba", ascending=False)
        row = {"point": i}
        if hit.empty:
            row.update(tile=None, seg_proba=np.nan, top_share_pct=np.nan)
            for r in radii:
                row[f"max_{r}m"] = np.nan
            rows.append(row)
            continue
        h = hit.iloc[0]
        tile = seg[seg["mine_id"] == h["mine_id"]]
        row["tile"] = h["mine_id"]
        row["seg_proba"] = float(h["dump_proba"])
        row["top_share_pct"] = float(100 * tile.loc[tile["dump_proba"] >= h["dump_proba"], "_area"].sum()
                                    / tile["_area"].sum())
        row["tile_max"] = float(tile["dump_proba"].max())
        for r in radii:
            near = tile[tile.geometry.intersects(p.buffer(r))]
            row[f"max_{r}m"] = float(near["dump_proba"].max()) if len(near) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    keep = [c for c in points.columns if c != "geometry"]
    out = pd.concat([points[keep].reset_index(drop=True), out.drop(columns="point")], axis=1)
    ll = points.to_crs(4326).geometry
    out["maps_url"] = [f"https://www.google.com/maps/search/?api=1&query={y:.6f},{x:.6f}"
                       for x, y in zip(ll.x, ll.y)]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Modellbewertung an Punkten ablesen.")
    p.add_argument("--segments", default=None,
                   help="segments_classified.gpkg eines Laufs (bei Läufen mit Zusatzkacheln: die nur "
                        "bewerteten Kacheln).")
    p.add_argument("--run", action="append", default=[], metavar="ORDNER",
                   help="Laufordner: ergänzt die ehrlichen Out-of-Fold-Bewertungen aller Segmente, "
                        "die mittrainiert wurden (oof_predictions.pkl + Polygone).")
    p.add_argument("--points", required=True, help="Punktdatei (GeoJSON/GPKG).")
    p.add_argument("--out", required=True, help="Ausgabe-CSV.")
    args = p.parse_args(argv)
    parts = []
    if args.segments:
        parts.append(gpd.read_file(args.segments)[["mine_id", "dump_proba", "geometry"]])
    for folder in args.run:
        from eval_segment_size import load_run
        oof, poly = load_run(folder)
        merged = poly.merge(oof[["mine_id", "segment_id", "dump_proba"]], on=["mine_id", "segment_id"])
        parts.append(merged[["mine_id", "dump_proba", "geometry"]])
    if not parts:
        p.error("--segments und/oder --run angeben")
    seg = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    out = score_points(seg, gpd.read_file(args.points))
    out.to_csv(args.out, index=False)
    print(out.drop(columns="maps_url").round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
