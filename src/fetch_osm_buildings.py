"""
OSM-Gebäudeumrisse pro Mine laden
====================================

Holt für jedes Minenbild in imagery_dir die Gebäude-Polygone (OSM-Tag
"building") innerhalb seiner Ausdehnung von OpenStreetMap (Overpass-API,
kostenlos, ohne Account) und speichert sie als GeoPackage im selben
Koordinatensystem wie die Mine.

Idee: ein Gebäudedach ist geometrisch klar (rechteckig) und hell/dunkel-
kontrastreich (Dach vs. Schlagschatten) - genau die Muster, die im Bild
selbst nicht zuverlässig von einem Reifenhaufen zu unterscheiden waren
(siehe README, shape_rectangularity/brightness_extreme_fraction). OSM-
Gebäudeumrisse sind dagegen eine direkte Ground-Truth-Aussage ("hier steht
ein Gebäude"), keine indirekte Vermutung aus Spektral-/Formmerkmalen.

WICHTIG: das ist unvalidiert, bis die Überlappung tatsächlich an den
echten 36 positiven Segmenten vs. falsch-positiven Segmenten geprüft
wurde (siehe README) - OSM-Gebäudedaten für abgelegene Industrieflächen
in der Atacama können lückenhaft sein.

Nutzung:
    python src/fetch_osm_buildings.py --imagery-dir data/imagery --out-dir data/osm_buildings
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
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "otr-obia-pipeline/1.0 (research script)"


def _element_to_polygon(el):
    """Baut aus einem Overpass "out geom"-Way/Relation-Element ein Shapely-
    Polygon. Relationen (aus mehreren Ways) werden vereinfacht als das
    Polygon ihres ersten Outer-Members behandelt - für Gebäude praktisch
    immer ausreichend, komplexe Multipolygon-Beziehungen sind selten."""
    if el["type"] == "way":
        coords = [(pt["lon"], pt["lat"]) for pt in el.get("geometry", [])]
    elif el["type"] == "relation":
        outer = next((m for m in el.get("members", []) if m.get("role") == "outer"), None)
        if outer is None or "geometry" not in outer:
            return None
        coords = [(pt["lon"], pt["lat"]) for pt in outer["geometry"]]
    else:
        return None
    if len(coords) < 4 or coords[0] != coords[-1]:
        return None
    poly = Polygon(coords)
    return poly if poly.is_valid and not poly.is_empty else None


def query_overpass(bbox_wgs84, timeout=60, retries=3):
    """bbox_wgs84: (south, west, north, east). Gibt eine Liste von Shapely-
    Polygonen zurück (leer, wenn keine Gebäude gefunden wurden).

    Nutzt requests mit explizitem User-Agent: der overpass.openstreetmap.fr-
    Mirror lehnt requests/urllib3-Anfragen mit 403 ("only available to
    white-listed usages") ab (vermutlich TLS-Fingerprinting gegen Bot-
    Clients), deshalb overpass-api.de (offizielle Hauptinstanz) statt-
    dessen - dort reicht ein gesetzter User-Agent."""
    south, west, north, east = bbox_wgs84
    query = (
        f"[out:json][timeout:{timeout}];"
        f'(way["building"]({south},{west},{north},{east});'
        f'relation["building"]({south},{west},{north},{east}););'
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
            polys = [p for p in (_element_to_polygon(e) for e in elements) if p is not None]
            return polys
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

    polys = query_overpass((south, west, north, east))
    if polys:
        gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_crs(crs)
    else:
        gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
    gdf.to_file(out_path, driver="GPKG")
    return f"{len(gdf)} Gebäude-Polygone"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lädt OSM-Gebäudeumrisse (Overpass-API) passend zu jedem Minenbild."
    )
    parser.add_argument("--imagery-dir", default="data/imagery", metavar="DIR",
                        help="Ordner mit den Sentinel-2-GeoTIFFs (ein Bild pro Mine).")
    parser.add_argument("--out-dir", default="data/osm_buildings", metavar="DIR",
                        help="Zielordner für die Gebäude-GeoPackages.")
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
