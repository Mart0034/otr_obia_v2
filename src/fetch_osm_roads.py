"""
OSM-Straßen pro Mine laden
============================

Holt für jedes Minenbild in imagery_dir die Straßen/Wege (OSM-Tag
"highway") innerhalb seiner Ausdehnung von OpenStreetMap (Overpass-API,
kostenlos, ohne Account) und speichert sie als GeoPackage (Linien) im
selben Koordinatensystem wie die Mine.

Idee: ein dichtes Gitter aus kurzen, meist parallelen/rechtwinkligen
Wegen ist die Signatur eines Fahrzeug-/Parkplatzbereichs einer
Industrieanlage - ein einzelner Zufahrtsweg zu einer echten Halde sieht
dagegen ganz anders aus (siehe README, "Straßen-Dichte als Merkmal").
Reine Nähe zu IRGENDEINER Straße trennt kaum (11% der echten Dumps liegen
selbst in 12m Nähe zu einer Straße), aber die lokale Straßen-DICHTE tut es
deutlich besser.

Nutzung:
    python src/fetch_osm_roads.py --imagery-dir data/imagery --out-dir data/osm_roads
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
from shapely.geometry import LineString

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "otr-obia-pipeline/1.0 (research script)"


def _element_to_line(el):
    if el["type"] != "way":
        return None
    coords = [(pt["lon"], pt["lat"]) for pt in el.get("geometry", [])]
    if len(coords) < 2:
        return None
    line = LineString(coords)
    return line if line.is_valid and not line.is_empty else None


def query_overpass(bbox_wgs84, timeout=60, retries=3):
    """bbox_wgs84: (south, west, north, east). Gibt eine Liste von Shapely-
    LineStrings zurück (leer, wenn keine Straßen gefunden wurden).

    Nutzt requests mit explizitem User-Agent - siehe fetch_osm_buildings.py
    für den Grund (403 gegen requests/urllib3 auf overpass.openstreetmap.fr,
    overpass-api.de mit User-Agent funktioniert)."""
    south, west, north, east = bbox_wgs84
    query = (
        f"[out:json][timeout:{timeout}];"
        f'(way["highway"]({south},{west},{north},{east}););'
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
            lines = [ln for ln in (_element_to_line(e) for e in elements) if ln is not None]
            return lines
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

    lines = query_overpass((south, west, north, east))
    if lines:
        gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326").to_crs(crs)
    else:
        gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
    gdf.to_file(out_path, driver="GPKG")
    return f"{len(gdf)} Straßen-Linien"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lädt OSM-Straßen (Overpass-API) passend zu jedem Minenbild."
    )
    parser.add_argument("--imagery-dir", default="data/imagery", metavar="DIR",
                        help="Ordner mit den Sentinel-2-GeoTIFFs (ein Bild pro Mine).")
    parser.add_argument("--out-dir", default="data/osm_roads", metavar="DIR",
                        help="Zielordner für die Straßen-GeoPackages.")
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
