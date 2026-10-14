"""
Beispiel-Daten für die Demo-Seite: Galerie echter Funde und die Vorher/Nachher-Karte einer Mine
=============================================================================================

Schreibt docs/data/examples.js (window.OTR_EX). Enthält Umrisse (Längen-/Breitengrad, vereinfacht), keine Standort-IDs.

  * "steps":   die drei Beispiele des Schritt-für-Schritt-Bereichs (Umriss des Fundes)
  * "gallery": weitere Funde mit Größe, KI-Wert und Art
  * "before":  Umgebung einer Mine: von Hand markierte Halden vom Anfang und alles, was das System seither gefunden hat

Nutzung:
    python scripts/export_site_examples.py --reviewed rN=datei.gpkg ... --known data/dump_labels.gpkg \\
        --steps S0203:cl_00025 S0025:q_way_849970931 S0057:q_way_884219252 \\
        --gallery S0002:q_way_330489641 ... --center -22.946,-69.085 --radius-m 3500
"""

import argparse
import json
import sys

import geopandas as gpd
import pandas as pd


def load_sites(specs):
    frames = []
    for item in specs:
        _, path = item.split("=", 1)
        g = gpd.read_file(path, layer="candidate_sites").to_crs(4326)
        for col in ("ai_dump_p", "ai_decision"):
            if col not in g.columns:
                g[col] = None
        frames.append(g)
    g = pd.concat(frames, ignore_index=True)
    g["review"] = g["review"].fillna("").astype(str).str.lower()
    return g[g["review"].isin(["dump", "partial"]) & (g["ai_decision"].fillna("") != "auto_clean")]


def geom_json(geom, tol=0.00001):
    return json.loads(gpd.GeoSeries([geom.simplify(tol)]).to_json())["features"][0]["geometry"]


def pick(sites, spec):
    out = []
    for s in spec:
        sid, mine = s.split(":")
        r = sites[(sites["site_id"] == sid) & (sites["mine_id"] == mine)].iloc[0]
        out.append({"geom": geom_json(r.geometry), "area": int(round(r["area_m2"], -2)),
                    "ai": None if pd.isna(r["ai_dump_p"]) else int(round(r["ai_dump_p"] * 100)), "kind": r["review"]})
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--reviewed", nargs="+", required=True)
    p.add_argument("--known", required=True)
    p.add_argument("--steps", nargs="+", required=True)
    p.add_argument("--gallery", nargs="+", required=True)
    p.add_argument("--center", required=True, help="lat,lon der Vorher/Nachher-Karte")
    p.add_argument("--radius-m", type=float, default=3500)
    p.add_argument("--out", default="docs/data/examples.js")
    args = p.parse_args(argv)
    sites = load_sites(args.reviewed)
    lat, lon = (float(x) for x in args.center.split(","))
    c = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(32719).iloc[0]
    known = gpd.read_file(args.known).to_crs(32719)
    kn = known[known.geometry.centroid.distance(c) <= args.radius_m]
    s32 = sites.to_crs(32719).copy()
    s32["_k"] = list(zip(s32.geometry.centroid.x.round(-1), s32.geometry.centroid.y.round(-1), s32["area_m2"].round(-1)))
    s32 = s32.drop_duplicates("_k")
    fn = s32[s32.geometry.centroid.distance(c) <= args.radius_m]
    to4326 = lambda g: g.to_crs(4326)
    out = {
        "steps": {k: v["geom"] for k, v in zip("abc", pick(sites, args.steps))},
        "gallery": pick(sites, args.gallery),
        "before": {"center": [lat, lon],
                   "known": [geom_json(g, 0.00002) for g in to4326(kn).geometry],
                   "found": [{"geom": geom_json(g, 0.00002), "kind": k} for g, k in zip(to4326(fn).geometry, fn["review"])]},
    }
    with open(args.out, "w") as f:
        f.write("window.OTR_EX=" + json.dumps(out, separators=(",", ":")) + ";\n")
    print(len(out["gallery"]), "gallery,", len(kn), "known,", len(fn), "found in the window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
