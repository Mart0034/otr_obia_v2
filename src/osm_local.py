"""
OSM-Daten für die Kacheln aus einer lokalen Länderdatei schneiden (statt Overpass)
==================================================================================

Die kostenlose Overpass-API ist oft überlastet (viele "504 Gateway Timeout"), und eine Abfrage pro Kachel und
Datenart dauert Minuten. Dieses Skript liest stattdessen EINMAL die ganze OSM-Datei eines Landes (z.B.
Geofabrik `chile-latest.osm.pbf`, ca. 350 MB, kostenlos), zieht Straßen, Gebäude und Punkte/Gewerbeflächen
heraus und schneidet daraus pro Kachel dieselben GeoPackages wie die fetch_osm_*-Skripte:

    <out-root>/osm_roads/<kachel>.gpkg      Straßenlinien (highway=*)
    <out-root>/osm_buildings/<kachel>.gpkg  Gebäudeumrisse (building=*)
    <out-root>/osm_poi/<kachel>.gpkg        amenity/shop/tourism-Punkte, amenity/shop-Flächen, landuse=retail/commercial
                                            (Spalte `kind`: "amenity/shop" oder "landuse")

Schritt 1 (einmal, ca. 10-20 Minuten): die Landesdatei in drei GeoPackages zerlegen (`--build-dir`).
Schritt 2 (Sekunden pro Kachel): pro Kachel zuschneiden. Vorhandene Dateien bleiben stehen (so füllt man nur
die Kacheln, bei denen Overpass fehlschlug); mit --overwrite wird alles neu geschrieben.

Nutzung:
    python src/osm_local.py --pbf data/osm/chile-latest.osm.pbf --build-dir data/osm_chile \\
        --imagery-dir data/chile_new/tiles/imagery --out-root data/chile_new/tiles
"""

import argparse
import glob
import logging
import os
import re
import sys

import geopandas as gpd
import pandas as pd
import pyogrio
import rasterio
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)

LAYERS = ("roads", "buildings", "poi")
_TAG = re.compile(r'"([^"=]+)"=>"((?:[^"\\]|\\.)*)"')


def parse_other_tags(text):
    """hstore-Text von GDAL ('"amenity"=>"cafe","name"=>"X"') -> dict."""
    if not isinstance(text, str):
        return {}
    return {k: v for k, v in _TAG.findall(text)}


def _tag_series(df, key):
    """Wert eines Tags: eigene Spalte (falls GDAL sie führt) oder aus other_tags."""
    if key in df.columns:
        col = df[key]
    else:
        col = pd.Series([None] * len(df), index=df.index)
    if "other_tags" in df.columns:
        extra = df["other_tags"].map(lambda t: parse_other_tags(t).get(key))
        col = col.where(col.notna(), extra)
    return col


def poi_kind(df):
    """Spalte `kind` für Punkte/Flächen: 'landuse' bei landuse=retail/commercial, 'amenity/shop' bei amenity/shop/
    tourism; sonst None (Zeile fällt weg)."""
    landuse = _tag_series(df, "landuse")
    amenity, shop, tourism = (_tag_series(df, k) for k in ("amenity", "shop", "tourism"))
    is_landuse = landuse.isin(["retail", "commercial"])
    is_shop = amenity.notna() | shop.notna() | tourism.notna()
    kind = pd.Series([None] * len(df), index=df.index, dtype=object)
    kind[is_shop] = "amenity/shop"
    kind[is_landuse] = "landuse"
    return kind


def _read(pbf, layer, **kw):
    return pyogrio.read_dataframe(pbf, layer=layer, **kw)


def build(pbf, build_dir, overwrite=False):
    """Landesdatei in roads.gpkg, buildings.gpkg, poi.gpkg zerlegen (EPSG:4326, mit Raumindex)."""
    os.makedirs(build_dir, exist_ok=True)
    paths = {k: os.path.join(build_dir, f"{k}.gpkg") for k in LAYERS}
    if not overwrite and all(os.path.exists(p) for p in paths.values()):
        logger.info("Zerlegte Dateien schon vorhanden (%s) - überspringe", build_dir)
        return paths

    logger.info("Straßen lesen ...")
    lines = _read(pbf, "lines", where="highway IS NOT NULL", columns=["highway"])
    lines[["geometry"]].to_file(paths["roads"], driver="GPKG")
    logger.info("  %d Straßenlinien", len(lines))
    del lines

    logger.info("Gebäude lesen ...")
    bld = _read(pbf, "multipolygons", where="building IS NOT NULL", columns=["building"])
    bld[["geometry"]].to_file(paths["buildings"], driver="GPKG")
    logger.info("  %d Gebäude", len(bld))
    del bld

    logger.info("Flächen lesen (Geschäfte, Gewerbe) ...")
    areas = _read(pbf, "multipolygons", columns=["amenity", "shop", "tourism", "landuse"],
                  where="amenity IS NOT NULL OR shop IS NOT NULL OR tourism IS NOT NULL OR landuse IN ('retail','commercial')")
    area_poi = areas.copy()
    area_poi["kind"] = poi_kind(areas)
    area_poi = area_poi[area_poi["kind"].notna()][["kind", "geometry"]]
    del areas

    logger.info("Punkte lesen (amenity/shop/tourism) ...")
    pts = _read(pbf, "points")
    pts["kind"] = poi_kind(pts)
    pts = pts[pts["kind"].notna()][["kind", "geometry"]]
    poi = pd.concat([area_poi, pts], ignore_index=True)
    gpd.GeoDataFrame(poi, geometry="geometry", crs="EPSG:4326").to_file(paths["poi"], driver="GPKG")
    logger.info("  %d Punkte/Flächen", len(poi))
    return paths


def clip_tile(paths, tif_path, out_root, overwrite=False):
    """Die drei Dateien für eine Kachel zuschneiden. Gibt zurück, wie viele Elemente je Art geschrieben wurden."""
    tile = os.path.splitext(os.path.basename(tif_path))[0]
    with rasterio.open(tif_path) as ref:
        crs, bounds = ref.crs, ref.bounds
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds)
    counts = {}
    for key, folder in (("roads", "osm_roads"), ("buildings", "osm_buildings"), ("poi", "osm_poi")):
        out = os.path.join(out_root, folder, tile + ".gpkg")
        if os.path.exists(out) and not overwrite:
            counts[key] = None
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        gdf = pyogrio.read_dataframe(paths[key], bbox=(west, south, east, north))
        if len(gdf):
            gdf = gdf.to_crs(crs)
        else:
            gdf = gpd.GeoDataFrame({"kind": []} if key == "poi" else {}, geometry=[], crs=crs)
        gdf.to_file(out, driver="GPKG")
        counts[key] = len(gdf)
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description="OSM-Straßen/Gebäude/Punkte aus einer lokalen Länderdatei pro Kachel schneiden.")
    p.add_argument("--pbf", help="Landesdatei (.osm.pbf), nur nötig, wenn --build-dir noch nicht gefüllt ist.")
    p.add_argument("--build-dir", default="data/osm_chile", help="Ordner für roads/buildings/poi.gpkg (wird einmal gebaut).")
    p.add_argument("--imagery-dir", help="Ordner mit den Kachel-GeoTIFFs.")
    p.add_argument("--out-root", help="Zielordner; darin osm_roads/, osm_buildings/, osm_poi/.")
    p.add_argument("--overwrite", action="store_true", help="Auch vorhandene Kachel-Dateien neu schreiben.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    paths = {k: os.path.join(args.build_dir, f"{k}.gpkg") for k in LAYERS}
    if not all(os.path.exists(v) for v in paths.values()):
        if not args.pbf:
            p.error("--pbf fehlt: die zerlegten Dateien sind noch nicht gebaut")
        paths = build(args.pbf, args.build_dir)
    if not (args.imagery_dir and args.out_root):
        return 0
    tifs = sorted(glob.glob(os.path.join(args.imagery_dir, "*.tif")))
    logger.info("%d Kacheln zuschneiden", len(tifs))
    for i, tif in enumerate(tifs, 1):
        counts = clip_tile(paths, tif, args.out_root, args.overwrite)
        logger.info("  [%d/%d] %s: %s", i, len(tifs), os.path.basename(tif),
                    ", ".join(f"{k} {'übersprungen' if v is None else v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
