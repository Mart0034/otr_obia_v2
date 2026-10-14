"""
Blindtest: wie viele bestätigte Halden findet das System in Kacheln, die es nie gesehen hat?
==========================================================================================

Die bestätigten Halden (von Hand markiert oder in den Prüfrunden bestätigt) werden nach Kacheln in K Gruppen
geteilt. In jedem Durchgang wird eine Gruppe "versteckt": ihre Kacheln und alle Kacheln, die sich mit ihnen überlappen,
werden samt Halden-Polygonen, Fehlalarmen und Ignorier-Zonen aus dem Training genommen (sie werden nur noch bewertet).
Danach wird mit dem Rest neu trainiert, alles bewertet, die Kandidatenlisten wie im Normalbetrieb gebaut, und gezählt,
wie viele der versteckten Halden-Orte in den Listen landen.

    python scripts/blind_test.py prep  --fold 0 --folds 3 --suffix all8 --out data/blind/f0
    python scripts/blind_test.py score --fold 0 --out data/blind/f0 --lists RUN/candidate_sites_a.gpkg RUN/candidate_sites_b.gpkg
    python scripts/blind_test.py merge --glob "data/blind/f*/score.json" --out blind_results.json
"""

import argparse
import glob
import json
import os
import random
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

OUT = "data/reviewed/out/"
Q = "data/sites_quarries"
TILE_DIRS = {"imagery": "data/imagery", "labeled": "data/run15/labeled",
             "reviewed": Q + "/imagery_reviewed", "extra": Q + "/imagery_rest"}
WORK = "EPSG:32719"


def footprints():
    rows = []
    for kind, d in TILE_DIRS.items():
        for f in sorted(glob.glob(os.path.join(d, "*.tif"))):
            with rasterio.open(f) as r:
                b = r.bounds
            rows.append({"tile": os.path.splitext(os.path.basename(f))[0], "kind": kind, "path": os.path.realpath(f),
                         "geometry": box(b.left, b.bottom, b.right, b.top)})
    return gpd.GeoDataFrame(rows, crs=WORK)


def truth_polygons(suffix):
    return gpd.read_file(f"{OUT}labels_{suffix}_for_eval.gpkg").to_crs(WORK)


def choose_folds(truth, n_folds, seed):
    """Kacheln mit Halden auf n_folds Gruppen verteilen, etwa gleich viele Halden-Polygone je Gruppe."""
    counts = truth.groupby("mine_id").size().sort_values(ascending=False)
    ids = list(counts.index)
    random.Random(seed).shuffle(ids)
    ids.sort(key=lambda i: -counts[i])
    load, groups = [0] * n_folds, [[] for _ in range(n_folds)]
    for i in ids:
        k = int(np.argmin(load)); groups[k].append(i); load[k] += int(counts[i])
    return groups


def link_tiles(fp, hidden, out):
    for kind in TILE_DIRS:
        os.makedirs(os.path.join(out, kind), exist_ok=True)
    for r in fp.itertuples():
        kind = "extra" if r.tile in hidden else r.kind
        dst = os.path.join(out, kind, r.tile + ".tif")
        if not os.path.lexists(dst):
            os.symlink(r.path, dst)


def prep(a):
    truth = truth_polygons(a.suffix)
    groups = choose_folds(truth, a.folds, a.seed)
    held = set(groups[a.fold])
    fp = footprints()
    held_geom = fp[fp["tile"].isin(held)].geometry.unary_union
    overlap = set(fp[fp.geometry.intersection(held_geom).area > 0]["tile"])        # auch überlappende Kacheln raus
    removed = overlap | held
    os.makedirs(a.out, exist_ok=True)
    link_tiles(fp, removed, a.out)

    def keep(path, name, to=None):
        g = gpd.read_file(path)
        n0 = len(g)
        g = g[~g["mine_id"].isin(removed)]
        dst = os.path.join(a.out, name)
        g.to_file(dst, driver="GeoJSON" if name.endswith(".geojson") else "GPKG")
        print(f"{name}: {n0} -> {len(g)}")

    keep("data/dump_labels.gpkg", "dump_labels.gpkg")
    keep("data/run15/labels_extra.gpkg", "labels_extra.gpkg")
    keep(f"{OUT}labels_reviewed_dump_{a.suffix}.gpkg", "labels_reviewed_dump.gpkg")
    keep(f"{OUT}labels_{a.suffix}_for_eval.gpkg", "labels_for_eval.gpkg")
    keep(f"{OUT}negatives_{a.suffix}_clean.gpkg", "negatives_clean.gpkg")
    keep(f"{OUT}ignore_reviewed_{a.suffix}.geojson", "ignore_reviewed.geojson")
    pts = gpd.read_file("data/run15/ignore_points.geojson").to_crs(WORK)
    pts = pts[~pts.within(fp[fp["tile"].isin(removed)].geometry.unary_union)].to_crs(4326)
    pts.to_file(os.path.join(a.out, "ignore_points.geojson"), driver="GeoJSON")
    held_truth = truth[truth["mine_id"].isin(held)]
    held_truth.to_file(os.path.join(a.out, "truth.gpkg"), driver="GPKG")
    json.dump({"fold": a.fold, "hidden_tiles": sorted(held), "removed_tiles": sorted(removed),
               "hidden_polygons": int(len(held_truth))}, open(os.path.join(a.out, "prep.json"), "w"))
    print(f"fold {a.fold}: {len(held)} hidden tiles, {len(removed)} removed from training, {len(held_truth)} hidden dump polygons")


def places(gdf, buf=100):
    u = gdf.geometry.buffer(buf).unary_union
    parts = list(u.geoms) if hasattr(u, "geoms") else [u]
    return parts


def score(a):
    truth = gpd.read_file(os.path.join(a.out, "truth.gpkg")).to_crs(WORK)
    tparts = places(truth)
    # ein Ort der Wahrheit gilt als gefunden, wenn ein Standort der Liste mindestens ein Polygon davon berührt
    res = {"fold": a.fold, "hidden_polygons": int(len(truth)), "hidden_places": len(tparts), "lists": {}}
    for path in a.lists:
        sites = gpd.read_file(path, layer="candidate_sites").to_crs(WORK)
        found = 0
        for part in tparts:
            polys = truth[truth.geometry.intersects(part)]
            if sites.geometry.intersects(polys.geometry.unary_union).any():
                found += 1
        res["lists"][os.path.basename(path)] = {"sites": int(len(sites)), "found_places": found,
                                                  "recall": round(found / max(len(tparts), 1), 4)}
    json.dump(res, open(os.path.join(a.out, "score.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def merge(a):
    parts = [json.load(open(f)) for f in sorted(glob.glob(a.glob))]
    out = {"folds": len(parts), "hidden_places": sum(p["hidden_places"] for p in parts),
           "hidden_polygons": sum(p["hidden_polygons"] for p in parts), "lists": {}}
    names = parts[0]["lists"].keys()
    for n in names:
        f = sum(p["lists"][n]["found_places"] for p in parts)
        out["lists"][n] = {"found_places": f, "recall": round(f / max(out["hidden_places"], 1), 4),
                           "sites_per_fold": [p["lists"][n]["sites"] for p in parts]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep"); s.add_argument("--fold", type=int, required=True); s.add_argument("--folds", type=int, default=3)
    s.add_argument("--suffix", default="all8"); s.add_argument("--seed", type=int, default=7); s.add_argument("--out", required=True)
    s.set_defaults(fn=prep)
    s = sub.add_parser("score"); s.add_argument("--fold", type=int, required=True); s.add_argument("--out", required=True)
    s.add_argument("--lists", nargs="+", required=True); s.set_defaults(fn=score)
    s = sub.add_parser("merge"); s.add_argument("--glob", required=True); s.add_argument("--out", required=True); s.set_defaults(fn=merge)
    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
