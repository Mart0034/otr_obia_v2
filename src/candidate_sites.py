"""
Kandidaten-Standorte aus den Segment-Vorhersagen
================================================

Das Modell bewertet einzelne Segmente. Für die Praxis (zu Fuß prüfen, in
hochauflösenden Bildern ansehen, Größe/Tonnage schätzen) sind aber ZUSAMMEN-
HÄNGENDE Funde interessant: ein Cluster aus 40 hoch bewerteten Segmenten ist
etwas anderes als ein einzelnes. Dieses Skript

  1. wählt die am höchsten bewerteten Segmente je Mine, bis --area-pct % der
     Minenfläche markiert sind (wie im Segmentgrößen-Vergleich) - oder, mit
     --threshold, alle Segmente mit dump_proba >= Schwelle,
  2. verbindet Segmente, die höchstens --join-dist-m auseinanderliegen, zu
     einem Standort,
  3. berechnet je Standort Fläche, mittlere/maximale Bewertung, einen Rang
     (Fläche x mittlere Bewertung) und einen Google-Maps-Link,
  4. schreibt alles als GeoPackage (für QGIS) und als CSV.

Die Spalten `review` und `note` bleiben leer - in QGIS kann man dort eintragen,
ob ein Fund eine echte Halde ist (dump), sauberes Gelände (clean) oder unklar
(unsure). Das ist die Grundlage, um echte Negativbeispiele zu sammeln.

Nutzung:
    python src/candidate_sites.py --segments out_px15/segments_classified.gpkg \\
        --labels data/dump_labels.gpkg --out out_px15/candidate_sites.gpkg
"""

import argparse
import logging
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Größenklassen nach zusammenhängender Fläche (m²). Zum Vergleich: die Kreis-
# Labels für mittlere/große/sehr große Halden haben ~5.000 / ~15.000 / ~70.000 m².
SIZE_CLASSES = ((2500, "small"), (10000, "medium"), (40000, "large"))
INFO_COLS = ("is_building", "is_poi", "is_road_grid")


def _size_class(area_m2):
    for limit, name in SIZE_CLASSES:
        if area_m2 < limit:
            return name
    return "very_large"


def select_seeds(seg, threshold=None, area_pct=2.0):
    """Segmente, die als Fund gelten: dump_proba >= threshold, oder (ohne
    threshold) je Mine die höchstbewerteten bis area_pct % der gültigen Fläche."""
    seg = seg[seg["dump_proba"].notna()].copy()
    seg["_area"] = seg.geometry.area
    if threshold is not None:
        return seg[seg["dump_proba"] >= threshold]
    keep = []
    for _, grp in seg.groupby("mine_id"):
        grp = grp.sort_values("dump_proba", ascending=False)
        keep.append(grp[grp["_area"].cumsum() - grp["_area"] < grp["_area"].sum() * area_pct / 100.0])
    return pd.concat(keep) if keep else seg.iloc[0:0]


def build_candidate_sites(seg, threshold=None, area_pct=2.0, join_dist_m=20.0, min_area_m2=0.0,
                          labels=None, known_dist_m=50.0, top=None):
    """seg: Segmente mit mine_id, dump_proba, geometry (projiziertes CRS in Metern).
    Gibt ein GeoDataFrame mit einem Standort pro Zeile zurück, nach Rang sortiert."""
    if seg.crs is None or not seg.crs.is_projected:
        raise ValueError("Die Segmente brauchen ein projiziertes CRS (Meter), z.B. UTM.")
    seeds = select_seeds(seg, threshold, area_pct)
    parts = []
    for mine, grp in seeds.groupby("mine_id"):
        merged = grp.geometry.buffer(join_dist_m / 2.0).union_all()
        comps = gpd.GeoDataFrame(geometry=list(getattr(merged, "geoms", [merged])), crs=seg.crs)
        comps["_comp"] = np.arange(len(comps))
        joined = gpd.sjoin(grp, comps, predicate="intersects", how="inner")
        joined = joined[~joined.index.duplicated(keep="first")]
        for comp, g in joined.groupby("_comp"):
            area = float(g["_area"].sum())
            row = {
                "mine_id": mine,
                "n_segments": int(len(g)),
                "area_m2": area,
                "mean_proba": float((g["dump_proba"] * g["_area"]).sum() / area),
                "max_proba": float(g["dump_proba"].max()),
                "geometry": g.geometry.union_all(),
            }
            for col in INFO_COLS:
                if col in g.columns:
                    row[f"frac_{col}"] = float((g[col].astype(float) * g["_area"]).sum() / area)
            parts.append(row)
    cols = ["mine_id", "n_segments", "area_m2", "mean_proba", "max_proba"]
    if not parts:
        return gpd.GeoDataFrame(columns=["site_id", "rank", *cols, "geometry"],
                                geometry="geometry", crs=seg.crs)
    sites = gpd.GeoDataFrame(parts, geometry="geometry", crs=seg.crs)
    sites = sites[sites["area_m2"] >= min_area_m2]
    sites["score"] = sites["area_m2"] * sites["mean_proba"]
    sites = sites.sort_values("score", ascending=False).reset_index(drop=True)
    if top:
        sites = sites.head(top)
    sites.insert(0, "rank", np.arange(1, len(sites) + 1))
    sites.insert(1, "site_id", [f"S{r:04d}" for r in sites["rank"]])
    sites["size_class"] = [_size_class(a) for a in sites["area_m2"]]

    pts = sites.geometry.representative_point()
    ll = gpd.GeoSeries(pts, crs=seg.crs).to_crs(4326)
    sites["lat"], sites["lon"] = ll.y.round(6).values, ll.x.round(6).values
    sites["maps_url"] = [
        f"https://www.google.com/maps/search/?api=1&query={la},{lo}"
        for la, lo in zip(sites["lat"], sites["lon"])
    ]

    if labels is not None and len(labels):
        lab = labels.to_crs(seg.crs)
        near = gpd.sjoin_nearest(sites[["geometry"]], lab[["geometry"]], how="left",
                                 distance_col="_d")
        d = near.groupby(level=0)["_d"].min()
        sites["near_known_dump"] = (d.reindex(sites.index) <= known_dist_m).values
    sites["review"] = ""
    sites["note"] = ""
    return sites


def main(argv=None):
    p = argparse.ArgumentParser(description="Kandidaten-Standorte aus Segment-Vorhersagen bauen.")
    p.add_argument("--segments", required=True, help="segments_classified.gpkg eines Laufs.")
    p.add_argument("--out", required=True, help="Ausgabe-GeoPackage (zusätzlich eine .csv daneben).")
    p.add_argument("--area-pct", type=float, default=2.0,
                   help="Je Mine die höchstbewerteten Segmente bis zu diesem %% der Fläche markieren "
                        "(Standard 2). Wird ignoriert, wenn --threshold gesetzt ist.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Stattdessen: alle Segmente mit dump_proba >= Schwelle markieren.")
    p.add_argument("--join-dist-m", type=float, default=20.0,
                   help="Segmente bis zu diesem Abstand gehören zum selben Standort.")
    p.add_argument("--min-area-m2", type=float, default=0.0, help="Kleinere Standorte weglassen.")
    p.add_argument("--top", type=int, default=None, help="Nur die besten N Standorte.")
    p.add_argument("--labels", default=None,
                   help="Bekannte Halden (GeoPackage); markiert Standorte nahe bekannter Halden.")
    p.add_argument("--known-dist-m", type=float, default=50.0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    seg = gpd.read_file(args.segments)
    labels = gpd.read_file(args.labels) if args.labels else None
    sites = build_candidate_sites(seg, args.threshold, args.area_pct, args.join_dist_m,
                                  args.min_area_m2, labels, args.known_dist_m, args.top)
    sites.to_file(args.out, layer="candidate_sites", driver="GPKG")
    csv_path = args.out.rsplit(".", 1)[0] + ".csv"
    pd.DataFrame(sites.drop(columns="geometry")).to_csv(csv_path, index=False)
    logging.info("%d Standorte -> %s (+ %s)", len(sites), args.out, csv_path)
    if len(sites):
        by = sites["size_class"].value_counts().to_dict()
        logging.info("Größenklassen: %s", by)
    return 0


if __name__ == "__main__":
    sys.exit(main())
