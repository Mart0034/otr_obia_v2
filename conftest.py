"""Shared pytest setup: makes src/ importable and provides a helper to
create small, made-up satellite images for tests, so tests don't need
real Sentinel-2 data."""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from otr_obia_pipeline import CONFIG  # noqa: E402


@pytest.fixture
def write_synthetic_raster():
    """Factory fixture: writes a small fake multi-band GeoTIFF at the given
    path and returns it. Used instead of real Sentinel-2 imagery in tests."""

    def _write(
        path,
        seed=0,
        crs="EPSG:32632",
        origin=(500000, 5600000),
        size=40,
        n_bands=len(CONFIG["band_names"]),
        nodata_corner=False,
        bright_patch=((5, 15), (5, 15)),
    ):
        rng = np.random.default_rng(seed)
        arr = rng.random((n_bands, size, size), dtype=np.float32) * 100
        (y0, y1), (x0, x1) = bright_patch
        arr[:, y0:y1, x0:x1] += 200  # a brighter "dump-like" patch
        if nodata_corner:
            arr[:, 0:5, 0:5] = -9999
        transform = from_origin(origin[0], origin[1], 10, 10)
        with rasterio.open(
            str(path),
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=n_bands,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(arr)
        return str(path)

    return _write
