"""
Sentinel-2-Basisbild für eine neue, noch nicht erfasste Stelle
=================================================================

Alle 138 vorhandenen Minen-Kacheln in data/imagery/ stammen aus "Stage 2"
des früheren Machbarkeitsstudien-Projekts, nicht aus diesem Repository -
siehe README. Dieses Skript baut denselben 10-Band-Stack (blue, green,
red, nir, swir1, swir2, ndvi, ndwi, dsi, bsi) frei nach Wahl für eine neue
Stelle (Mittelpunkt + Radius), damit auch Orte ohne vorhandene Kachel mit
der Pipeline durchsucht werden können.

WICHTIGE EINSCHRÄNKUNG: blue/green/red/nir/swir1/swir2 sind normale
Sentinel-2-L2A-Reflektanzwerte, und ndvi/ndwi/bsi wurden anhand der echten
Stage-2-Kacheln exakt zurückgerechnet (siehe unten, Fehler < 1e-7):

    ndvi = (nir - red) / (nir + red)
    ndwi = (green - swir1) / (green + swir1)         [modifiziertes NDWI]
    bsi  = ((swir1+red) - (nir+blue)) / ((swir1+red) + (nir+blue))

    dsi  = -(red + nir + swir1) / 3

DSI stand zunächst als unbekannt da: keine normalisierte Differenz traf die
echten Werte, weil DSI gar keine Verhältnisformel ist, sondern der negative
Mittelwert dreier Rohbänder. Gefunden per linearer Regression von DSI auf die
6 Rohbänder (R^2 = 1.000, Koeffizienten exakt -1/3 für red, nir und swir1,
0 für alle anderen, max. Fehler 4e-8 = float32-Genauigkeit über Stichproben
aus 23 Kacheln). Damit sind dsi_mean/dsi_std und die GLCM-Textur-Merkmale
neuer Stellen direkt mit den 138 bekannten Minen vergleichbar.

Nutzung:
    python src/fetch_s2_base_imagery.py --center-x 365950.1 --center-y 7367190.3 \
        --radius-m 1500 --mine-id new_001 --out-dir data/new_sites/imagery
"""

import argparse
import logging
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from fetch_s2_timeseries import _dedupe_by_date, to_reflectance, valid_mask
from fetch_sentinel1 import _read_on_grid, open_catalog, select_evenly

logger = logging.getLogger(__name__)

COLLECTION = "sentinel-2-l2a"
RAW_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
BAND_NAMES = ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "ndwi", "dsi", "bsi"]
PIXEL_SIZE_M = 10.0


def build_grid(center_x, center_y, radius_m, crs, pixel_size=PIXEL_SIZE_M):
    """Quadratisches Zielraster (2*radius_m Kantenlänge) um den Mittelpunkt,
    Pixelgröße wie die vorhandenen Minen-Kacheln (10m)."""
    half = radius_m
    width = height = int(round(2 * half / pixel_size))
    transform = from_origin(center_x - half, center_y + half, pixel_size, pixel_size)
    return transform, width, height


def compose_bands(catalog, crs, transform, width, height, start, end, max_scenes, max_cloud, min_valid):
    bbox = transform_bounds(crs, "EPSG:4326", *rasterio.transform.array_bounds(height, width, transform))
    items = [
        it for it in catalog.search(
            collections=[COLLECTION], bbox=bbox, datetime=f"{start}/{end}",
        ).items()
        if it.properties.get("eo:cloud_cover", 100) <= max_cloud
    ]
    chosen = select_evenly(_dedupe_by_date(items), max_scenes)
    if len(chosen) < min_valid:
        raise RuntimeError(f"nur {len(chosen)} wolkenarme Aufnahmen gefunden (mind. {min_valid})")

    def read(it, band, resampling):
        return _read_on_grid(it.assets[band].href, crs, transform, width, height, resampling)

    per_band_scenes = {b: [] for b in RAW_BANDS}
    for it in chosen:
        baseline = it.properties.get("s2:processing_baseline", "04.00")
        scl = read(it, "SCL", Resampling.nearest)
        mask = valid_mask(scl)
        for b in RAW_BANDS:
            refl = to_reflectance(read(it, b, Resampling.average), baseline)
            refl[~mask] = np.nan
            per_band_scenes[b].append(refl)

    with np.errstate(all="ignore"):
        composite = {b: np.nanmedian(np.stack(scenes), axis=0) for b, scenes in per_band_scenes.items()}
    months = sorted({it.datetime.month for it in chosen})
    return composite, len(chosen), months


def compute_dsi(red, nir, swir1):
    """DSI = -(red + nir + swir1) / 3, exakt aus den echten Stage-2-Kacheln
    zurückgerechnet (siehe Modul-Docstring)."""
    return -(red + nir + swir1) / 3


def compute_indices(blue, green, red, nir, swir1, swir2):
    """NDVI/NDWI/BSI, exakt zurückgerechnet aus den echten Stage-2-Kacheln
    (Fehler < 1e-7, siehe Modul-Docstring). DSI: siehe compute_dsi()."""
    with np.errstate(all="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndwi = (green - swir1) / (green + swir1)
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
    return ndvi, ndwi, bsi


def fetch_for_site(center_x, center_y, radius_m, out_path, catalog, crs="EPSG:32719",
                   start="2023-01-01", end="2023-12-31", max_scenes=8, max_cloud=20, min_valid=3):
    transform, width, height = build_grid(center_x, center_y, radius_m, crs)
    composite, n_scenes, months = compose_bands(
        catalog, crs, transform, width, height, start, end, max_scenes, max_cloud, min_valid
    )

    blue, green, red, nir, swir1, swir2 = (
        composite["B02"], composite["B03"], composite["B04"],
        composite["B08"], composite["B11"], composite["B12"],
    )
    ndvi, ndwi, bsi = compute_indices(blue, green, red, nir, swir1, swir2)
    dsi = compute_dsi(red, nir, swir1)

    stack = np.stack([blue, green, red, nir, swir1, swir2, ndvi, ndwi, dsi, bsi])

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": len(BAND_NAMES),
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack.astype(np.float32))
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(i, name)

    return f"{n_scenes} Aufnahmen (Monate {months}), {width}x{height} Pixel"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Baut ein 10-Band-Sentinel-2-Basisbild fuer eine neue Stelle "
                    "(Mittelpunkt + Radius), im selben Format wie data/imagery/*.tif."
    )
    parser.add_argument("--center-x", type=float, required=True, help="Mittelpunkt X (EPSG:32719).")
    parser.add_argument("--center-y", type=float, required=True, help="Mittelpunkt Y (EPSG:32719).")
    parser.add_argument("--radius-m", type=float, required=True,
                        help="Halbe Kantenlaenge des Zielgebiets in Metern.")
    parser.add_argument("--mine-id", required=True, help="Name fuer die Ausgabedatei (<mine-id>.tif).")
    parser.add_argument("--out-dir", default="data/new_sites/imagery", metavar="DIR")
    parser.add_argument("--crs", default="EPSG:32719")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--max-scenes", type=int, default=8)
    parser.add_argument("--max-cloud", type=float, default=20)
    parser.add_argument("--min-valid", type=int, default=3)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.mine_id}.tif")

    catalog = open_catalog()
    try:
        msg = fetch_for_site(
            args.center_x, args.center_y, args.radius_m, out_path, catalog, crs=args.crs,
            start=args.start, end=args.end, max_scenes=args.max_scenes,
            max_cloud=args.max_cloud, min_valid=args.min_valid,
        )
        logger.info("%s: %s", args.mine_id, msg)
    except Exception as e:
        logger.error("%s: fehlgeschlagen (%s)", args.mine_id, e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
