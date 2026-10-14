"""
Orte außerhalb von Minen aus OSM: Deponien, Recyclinghöfe, Reifenhändler, Werke, Industrieflächen
================================================================================================

Die bisherigen Läufe suchten um Minen und Steinbrüche. Reifenhalden können auch dort liegen, wo Reifen gesammelt oder
verarbeitet werden. Dieses Skript liest die Landesdatei (Geofabrik .osm.pbf) und macht daraus eine Vektordatei, die
sites_from_vector.py zu Kacheln machen kann:

  Art (kind), in dieser Reihenfolge der Wichtigkeit:
    waste       Deponie (landuse=landfill), Müllannahme/Recycling (amenity=waste_disposal/recycling/waste_transfer_station)
    tyres       Reifenhändler/-werkstatt (shop=tyres), Punkte werden zu Flächen (--point-buffer-m)
    works       Werke (man_made=works)
    industrial  Industrieflächen (landuse=industrial) ab --min-area-m2
    brownfield  Brachflächen (landuse=brownfield) ab --min-area-m2

  * nur zwischen --south und --north (Breitengrad), Koordinatensystem EPSG:32719 (UTM 19S, Chile),
  * ohne Orte, deren Mittelpunkt in einer schon vorhandenen Kachel liegt (--exclude-tiles-dir),
  * sortiert nach Art und dann Fläche, die ersten --limit Orte werden geschrieben.

Nutzung:
    python src/osm_places.py --pbf data/osm/chile-latest.osm.pbf --south -33 --north -17.5 \\
        --exclude-tiles-dir data/imagery --exclude-tiles-dir data/sites_quarries/imagery --out data/industrial/places.gpkg
"""

import argparse
import glob
import logging
import os
import sys

import geopandas as gpd
import pandas as pd
import pyogrio
import rasterio
from shapely.geometry import box

logger = logging.getLogger(__name__)
WORK_CRS = "EPSG:32719"
KIND_ORDER = ["waste", "tyres", "works", "industrial", "brownfield"]
POLY_WHERE = ("landuse IN ('landfill','industrial','brownfield') OR man_made = 'works' OR "
              "amenity IN ('waste_disposal','recycling','waste_transfer_station') OR shop = 'tyres'")
POINT_WHERE = "amenity IN ('waste_disposal','recycling','waste_transfer_station') OR shop = 'tyres' OR man_made = 'works'"


def classify(df):
    """Art je Zeile (siehe Modul-Text); Zeilen ohne passende Art bekommen None."""
    def col(name):
        return df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index)
    landuse, man_made, amenity, shop = col("landuse"), col("man_made"), col("amenity"), col("shop")
    kind = pd.Series([None] * len(df), index=df.index, dtype=object)
    kind[landuse == "brownfield"] = "brownfield"
    kind[landuse == "industrial"] = "industrial"
    kind[man_made == "works"] = "works"
    kind[shop == "tyres"] = "tyres"
    kind[amenity.isin(["waste_disposal", "recycling", "waste_transfer_station"]) | (landuse == "landfill")] = "waste"
    return kind


def build(gdf_polys, gdf_points, point_buffer_m=300.0, min_area_m2=5000.0):
    """Flächen und gepufferte Punkte (beide schon in EPSG:32719) zu einer Tabelle (kind, area_m2, geometry) zusammenführen."""
    parts = []
    for g, is_point in ((gdf_polys, False), (gdf_points, True)):
        if g is None or not len(g):
            continue
        g = g.copy()
        g["kind"] = classify(g)
        g = g[g["kind"].notna()]
        g["geometry"] = g.geometry.buffer(point_buffer_m) if is_point else g.geometry
        parts.append(g[["kind", "geometry"]])
    if not parts:
        return gpd.GeoDataFrame({"kind": [], "area_m2": []}, geometry=[], crs=WORK_CRS)
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=WORK_CRS)
    out["area_m2"] = out.geometry.area
    small = out["kind"].isin(["industrial", "brownfield"]) & (out["area_m2"] < min_area_m2)
    out = out[~small].copy()
    out["rank"] = out["kind"].map({k: i for i, k in enumerate(KIND_ORDER)})
    return out.sort_values(["rank", "area_m2"], ascending=[True, False]).drop(columns="rank").reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Deponien, Recyclinghöfe, Reifenhändler, Werke und Industrieflächen aus OSM als Orte.")
    p.add_argument("--pbf", required=True)
    p.add_argument("--south", type=float, default=-33.0)
    p.add_argument("--north", type=float, default=-17.5)
    p.add_argument("--west", type=float, default=-76.0)
    p.add_argument("--east", type=float, default=-66.0)
    p.add_argument("--min-area-m2", type=float, default=5000.0, help="Kleinste Fläche für Industrie/Brachflächen.")
    p.add_argument("--point-buffer-m", type=float, default=300.0)
    p.add_argument("--exclude-tiles-dir", action="append", default=[])
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    bbox = (a.west, a.south, a.east, a.north)
    logger.info("Flächen lesen ...")
    polys = pyogrio.read_dataframe(a.pbf, layer="multipolygons", where=POLY_WHERE, bbox=bbox)
    logger.info("  %d Flächen", len(polys))
    logger.info("Punkte lesen ...")
    pts = pyogrio.read_dataframe(a.pbf, layer="points", where=POINT_WHERE, bbox=bbox)
    logger.info("  %d Punkte", len(pts))
    places = build(polys.to_crs(WORK_CRS) if len(polys) else None, pts.to_crs(WORK_CRS) if len(pts) else None,
                   a.point_buffer_m, a.min_area_m2)
    logger.info("%d Orte vor dem Aussortieren: %s", len(places), places["kind"].value_counts().to_dict())
    zones = []
    for d in a.exclude_tiles_dir:
        for tif in glob.glob(os.path.join(d, "*.tif")):
            with rasterio.open(tif) as ds:
                zones.append(gpd.GeoSeries([box(*ds.bounds)], crs=ds.crs).to_crs(WORK_CRS).iloc[0])
    if zones and len(places):
        zone = gpd.GeoSeries(zones, crs=WORK_CRS).union_all()
        places = places[~places.geometry.representative_point().within(zone)].reset_index(drop=True)
    places = places.head(a.limit)
    logger.info("%d Orte geschrieben: %s", len(places), places["kind"].value_counts().to_dict())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    places.to_file(a.out, driver="GPKG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
