"""
Vektordatei für "den Rest eines Landes" bauen (Minen-Umrisse, die noch nicht abgedeckt sind)
==============================================================================================

Nimmt die weltweiten Bergbau-Polygone (Tang & Werner 2023, Zenodo) und macht daraus die Orte, die man mit
sites_from_vector.py zu Kacheln machen kann:

  * nur Polygone innerhalb der Landesgrenze (Natural Earth, wird geladen),
  * ohne Polygone nahe den schon bearbeiteten Minen (--exclude-near) und ohne Orte, die schon als Kachel
    vorliegen (--exclude-tiles-dir),
  * Polygone ab --min-area-m2; sehr große Polygone werden in ein Raster zerlegt (--piece-m), damit auch die
    größten Minen als Kacheln (<= --max-tile-m) herauskommen statt als "Kachel zu groß" übersprungen zu werden,
  * absteigend nach Fläche sortiert (die ersten N sind "die größten Orte").

Zusätzlich (--ops-out) alle Polygone des Landes als Umrisse für die Kontext-Spalte (Mine/Steinbruch/Wüste).

Nutzung:
    python src/region_vector.py --polys data/chile_new/tang/74548_projected_polygons.shp --country Chile \\
        --exclude-near data/mine_boundaries.gpkg --exclude-tiles-dir data/chile_new/tiles/imagery \\
        --out data/chile_rest/places.gpkg --ops-out data/chile_rest/ops.gpkg
"""

import argparse
import glob
import io
import logging
import os
import sys
import zipfile

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from shapely.geometry import box

logger = logging.getLogger(__name__)
WORK_CRS = "EPSG:32719"
BORDER_URLS = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip",
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
)


def country_border(name, cache_dir):
    """Landesgrenze (EPSG:4326) aus Natural Earth; wird einmal geladen."""
    os.makedirs(cache_dir, exist_ok=True)
    shp = os.path.join(cache_dir, "ne_10m_admin_0_countries.shp")
    if not os.path.exists(shp):
        last = None
        for url in BORDER_URLS:
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                zipfile.ZipFile(io.BytesIO(r.content)).extractall(cache_dir)
                break
            except Exception as e:                      # nächste Quelle versuchen
                last = e
        else:
            raise RuntimeError(f"Landesgrenze nicht ladbar: {last}")
    world = gpd.read_file(shp)
    col = next(c for c in ("NAME", "ADMIN", "NAME_EN") if c in world.columns)
    sel = world[world[col].str.lower() == name.lower()]
    if sel.empty:
        raise ValueError(f"Land {name!r} nicht gefunden")
    return sel.to_crs(4326)


def split_big(gdf, piece_m, big_side_m):
    """Polygone, deren längere Seite > big_side_m ist, in Raster-Stücke (piece_m) zerlegen; Reste unter 5 % der
    Stückfläche fallen weg. Kleine Polygone bleiben unverändert."""
    rows = []
    for _, r in gdf.iterrows():
        x0, y0, x1, y1 = r.geometry.bounds
        if max(x1 - x0, y1 - y0) <= big_side_m:
            rows.append({"area_m2": r.geometry.area, "geometry": r.geometry})
            continue
        gx0, gy0 = int(x0 // piece_m), int(y0 // piece_m)
        gx1, gy1 = int(x1 // piece_m), int(y1 // piece_m)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                part = r.geometry.intersection(box(gx * piece_m, gy * piece_m, (gx + 1) * piece_m, (gy + 1) * piece_m))
                if not part.is_empty and part.area >= 0.05 * piece_m * piece_m:
                    rows.append({"area_m2": part.area, "geometry": part})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def main(argv=None):
    p = argparse.ArgumentParser(description="Orte für den Rest eines Landes aus den Bergbau-Polygonen bauen.")
    p.add_argument("--polys", required=True, help="Bergbau-Polygone (Shapefile, Tang & Werner).")
    p.add_argument("--country", default="Chile")
    p.add_argument("--border-cache", default="data/natural_earth")
    p.add_argument("--exclude-near", action="append", default=[], help="Vektordatei; Orte näher als --exclude-buffer-m fallen weg.")
    p.add_argument("--exclude-buffer-m", type=float, default=3000.0)
    p.add_argument("--exclude-tiles-dir", action="append", default=[], help="Ordner mit Kachel-GeoTIFFs, die schon vorliegen.")
    p.add_argument("--min-area-m2", type=float, default=150000.0)
    p.add_argument("--piece-m", type=float, default=2700.0)
    p.add_argument("--big-side-m", type=float, default=3500.0)
    p.add_argument("--out", required=True)
    p.add_argument("--ops-out")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    border = country_border(args.country, args.border_cache)
    crs = gpd.read_file(args.polys, rows=1).crs
    bbox = tuple(border.to_crs(crs).total_bounds)
    polys = gpd.read_file(args.polys, bbox=bbox).to_crs(WORK_CRS)
    border_utm = border.to_crs(WORK_CRS).geometry.union_all()
    polys = polys[polys.intersects(border_utm)].copy()
    logger.info("%d Bergbau-Polygone in %s.", len(polys), args.country)
    if args.ops_out:
        os.makedirs(os.path.dirname(args.ops_out) or ".", exist_ok=True)
        polys[["geometry"]].to_file(args.ops_out, driver="GPKG")

    zones = []
    for f in args.exclude_near:
        zones.append(gpd.read_file(f).to_crs(WORK_CRS).geometry.buffer(args.exclude_buffer_m).union_all())
    for d in args.exclude_tiles_dir:
        for tif in glob.glob(os.path.join(d, "*.tif")):
            with rasterio.open(tif) as ds:
                zones.append(gpd.GeoSeries([box(*ds.bounds)], crs=ds.crs).to_crs(WORK_CRS).iloc[0])
    if zones:
        zone = gpd.GeoSeries(zones, crs=WORK_CRS).union_all()
        polys = polys[~polys.intersects(zone)].copy()
    logger.info("%d Polygone außerhalb der bereits bearbeiteten Gebiete.", len(polys))
    polys = polys[polys.area >= args.min_area_m2].copy()
    logger.info("%d Polygone mit mindestens %.2f km².", len(polys), args.min_area_m2 / 1e6)
    places = split_big(polys, args.piece_m, args.big_side_m).sort_values("area_m2", ascending=False)
    places = places.reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    places.to_file(args.out, driver="GPKG")
    logger.info("%d Orte (Stücke großer Polygone eingerechnet), größter %.1f km², 200. %.2f km².", len(places),
                places.area_m2.max() / 1e6 if len(places) else 0,
                places.area_m2.iloc[min(199, len(places) - 1)] / 1e6 if len(places) else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
