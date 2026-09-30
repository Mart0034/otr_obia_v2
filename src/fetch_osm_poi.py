"""
OSM-Punkte und Gewerbeflächen pro Mine laden
==============================================

Holt für jedes Minenbild in imagery_dir OSM-Punkte (amenity, shop, tourism -
z.B. Restaurants, Tankstellen) sowie geschlossene amenity/shop-Flächen und
landuse=retail/commercial-Polygone und speichert sie als GeoPackage im selben
Koordinatensystem wie die Mine.

Grund: fetch_osm_buildings.py findet nur building=*-Umrisse. An Orten wie La
Negra ist ein Restaurant oft nur als Punkt kartiert, sein Gebäude fehlt - für
is_building unsichtbar. Dieser Datensatz füllt die Lücke (Spalte is_poi).

Nutzung:
    python src/fetch_osm_poi.py --imagery-dir data/imagery --out-dir data/osm_poi
"""

import argparse
import glob
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import rasterio
import requests
from rasterio.warp import transform_bounds
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "otr-obia-pipeline/1.0 (research script)"


def _element_to_geometry(el):
    """Overpass "out geom"-Element -> Shapely-Punkt bzw. geschlossenes Polygon
    (mit "kind"-Beschriftung), sonst None."""
    tags = el.get("tags", {})
    kind = "landuse" if "landuse" in tags else "amenity/shop"
    if el["type"] == "node":
        return Point(el["lon"], el["lat"]), kind
    if el["type"] == "way":
        coords = [(pt["lon"], pt["lat"]) for pt in el.get("geometry", [])]
        if len(coords) >= 4 and coords[0] == coords[-1]:
            poly = Polygon(coords)
            if poly.is_valid and not poly.is_empty:
                return poly, kind
    return None


def query_overpass(bbox_wgs84, timeout=60, retries=3):
    """bbox_wgs84: (south, west, north, east). Gibt eine Liste von
    (Shapely-Geometrie, kind)-Paaren zurück (leer, wenn nichts gefunden wurde).

    Nutzt requests mit explizitem User-Agent: der overpass.openstreetmap.fr-
    Mirror lehnt requests/urllib3-Anfragen mit 403 ("only available to
    white-listed usages") ab (vermutlich TLS-Fingerprinting gegen Bot-
    Clients), deshalb overpass-api.de (offizielle Hauptinstanz) statt-
    dessen - dort reicht ein gesetzter User-Agent."""
    south, west, north, east = bbox_wgs84
    bb = f"({south},{west},{north},{east})"
    query = (
        f"[out:json][timeout:{timeout}];"
        f'(node["amenity"]{bb};node["shop"]{bb};node["tourism"]{bb};'
        f'way["landuse"~"^(retail|commercial)$"]{bb};way["amenity"]{bb};way["shop"]{bb};);'
        f"out geom;"
    )
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.post(
                OVERPASS_URL, data={"data": query},
                headers={"User-Agent": USER_AGENT}, timeout=timeout + 10,
            )
            r.raise_for_status()
            data = r.json()
            elements = data.get("elements", [])
            return [g for g in (_element_to_geometry(e) for e in elements) if g is not None]
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Overpass-Abfrage fehlgeschlagen nach {retries} Versuchen: {last_exc}")


def fetch_for_mine(s2_path, out_path):
    with rasterio.open(s2_path) as ref:
        crs = ref.crs
        bounds = ref.bounds
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds)

    items = query_overpass((south, west, north, east))
    if items:
        gdf = gpd.GeoDataFrame(
            {"kind": [k for _, k in items]}, geometry=[g for g, _ in items], crs="EPSG:4326"
        ).to_crs(crs)
    else:
        gdf = gpd.GeoDataFrame({"kind": []}, geometry=[], crs=crs)
    gdf.to_file(out_path, driver="GPKG")
    return f"{len(gdf)} Punkte/Flächen"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lädt OSM-Punkte/Gewerbeflächen (Overpass-API) passend zu jedem Minenbild."
    )
    parser.add_argument("--imagery-dir", default="data/imagery", metavar="DIR",
                        help="Ordner mit den Sentinel-2-GeoTIFFs (ein Bild pro Mine).")
    parser.add_argument("--out-dir", default="data/osm_poi", metavar="DIR",
                        help="Zielordner für die POI-GeoPackages.")
    parser.add_argument("--n-workers", type=int, default=2,
                        help="Anzahl gleichzeitiger Overpass-Abfragen (niedrig halten, "
                             "um den kostenlosen Server nicht zu überlasten).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bereits vorhandene Dateien neu erzeugen.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.imagery_dir, "*.tif")))
    todo = []
    for p in paths:
        out = os.path.join(args.out_dir, os.path.basename(p).replace(".tif", ".gpkg"))
        if os.path.exists(out) and not args.overwrite:
            continue
        todo.append((p, out))
    logger.info("%d Minenbilder gefunden, %d müssen noch geladen werden.", len(paths), len(todo))

    failed = []
    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(fetch_for_mine, p, out): p for p, out in todo}
        for fut in as_completed(futures):
            name = os.path.basename(futures[fut])
            try:
                logger.info("  %s: %s", name, fut.result())
            except Exception as e:
                logger.error("  %s: fehlgeschlagen (%s)", name, e)
                failed.append(name)

    if failed:
        logger.warning("%d Minen fehlgeschlagen: %s", len(failed), failed)
    return 1 if failed and len(failed) == len(todo) else 0


if __name__ == "__main__":
    sys.exit(main())
