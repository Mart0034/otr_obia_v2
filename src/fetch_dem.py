"""
Höhenmodell (Copernicus DEM) pro Mine laden
============================================

Holt für jedes Sentinel-2-Minenbild das Copernicus-Höhenmodell GLO-30
(30m, weltweit) von Microsoft Planetary Computer - kostenlos, ohne Account.
Die Höhe wird bilinear auf EXAKT dasselbe Pixelraster wie das Sentinel-2-Bild
gebracht und als GeoTIFF mit gleichem Dateinamen in out_dir gespeichert.
Liegt eine Mine über mehreren DEM-Kacheln (1°x1°), werden diese
zusammengesetzt.

Die Pipeline (--dem-dir) berechnet daraus Hangneigung, Lage relativ zur
Umgebung und Hangausrichtung pro Segment (siehe terrain_features() in
otr_obia_pipeline.py) - gedacht gegen Falsch-Positive an Berghängen, deren
Schatten im optischen Bild ähnlich dunkel und unruhig aussehen wie eine
Reifenhalde.

Nutzung:
    python src/fetch_dem.py --imagery-dir data/imagery --out-dir data/dem
"""

import argparse
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from fetch_sentinel1 import _read_on_grid, open_catalog, run_for_all_mines

COLLECTION = "cop-dem-glo-30"


def mosaic_tiles(tiles):
    """Setzt mehrere aufs Zielraster gebrachte DEM-Kacheln zusammen: jede
    Kachel deckt nur ihren Teil ab (Rest NaN), an den Kachelgrenzen sind die
    Werte identisch -> Mittelwert über die vorhandenen Werte."""
    stack = np.stack(tiles).astype(np.float32)
    with np.errstate(all="ignore"):
        valid = np.isfinite(stack).any(axis=0)
        out = np.full(stack.shape[1:], np.nan, dtype=np.float32)
        out[valid] = np.nanmean(stack[:, valid], axis=0)
    return out


def fetch_for_mine(s2_path, out_path, catalog):
    with rasterio.open(s2_path) as ref:
        crs, transform = ref.crs, ref.transform
        width, height = ref.width, ref.height
        bbox = transform_bounds(crs, "EPSG:4326", *ref.bounds)

    items = list(catalog.search(collections=[COLLECTION], bbox=bbox).items())
    if not items:
        raise RuntimeError("keine DEM-Kachel gefunden")

    elevation = mosaic_tiles([
        _read_on_grid(it.assets["data"].href, crs, transform, width, height,
                      resampling=Resampling.bilinear)
        for it in items
    ])

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(elevation[np.newaxis])
        dst.set_band_description(1, "elevation_m")
    return f"{len(items)} DEM-Kachel(n), Höhe {np.nanmin(elevation):.0f}-{np.nanmax(elevation):.0f} m"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lädt das Copernicus-Höhenmodell passend zu jedem Sentinel-2-Minenbild."
    )
    parser.add_argument("--imagery-dir", default="data/imagery", metavar="DIR",
                        help="Ordner mit den Sentinel-2-GeoTIFFs (ein Bild pro Mine).")
    parser.add_argument("--out-dir", default="data/dem", metavar="DIR",
                        help="Zielordner für die DEM-GeoTIFFs.")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Anzahl Minen, die gleichzeitig heruntergeladen werden.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bereits vorhandene Dateien neu erzeugen.")
    args = parser.parse_args(argv)

    catalog = open_catalog()
    return run_for_all_mines(
        args.imagery_dir, args.out_dir, args.n_workers, args.overwrite,
        lambda s2, out: fetch_for_mine(s2, out, catalog),
    )


if __name__ == "__main__":
    sys.exit(main())
