"""
Mehrjahres-Merkmale pro Mine: wächst da etwas Dunkles?
======================================================

Eine Reifenhalde wächst: Wo vor Jahren nackter Boden war, liegen heute dunkle
Reifen, und die Fläche wird von Jahr zu Jahr größer. Die meisten Fehlalarme
(Straßen, Gesteinshalden, Gebäude, Boden) sind dagegen über die Jahre gleich
oder wechseln. Dieses Skript macht aus Sentinel-2 (L2A, Planetary Computer,
kostenlos, ohne Account) pro Jahr ein wolkenfreies Helligkeitsbild und daraus
sechs Merkmale pro Pixel:

  - s2my_bright_early:  Helligkeit in den ersten beiden Jahren mit Daten
  - s2my_bright_late:   Helligkeit in den letzten beiden Jahren mit Daten
  - s2my_bright_delta:  spät minus früh (negativ = dunkler geworden)
  - s2my_bright_slope:  Steigung der Helligkeit pro Jahr (kleinste Quadrate)
  - s2my_dark_years_frac: Anteil der Jahre, in denen das Pixel dunkel war
  - s2my_dark_onset:    wann es erstmals dunkel war (0 = erstes Jahr, 1 = letztes;
                        leer, wenn nie dunkel)

Pixel mit weniger als --min-years Jahren mit gültigen Daten bleiben leer (NaN).

Nutzung:
    python src/fetch_s2_multiyear.py --imagery-dir data/imagery --out-dir data/s2_multiyear
"""

import argparse
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from fetch_s2_timeseries import BRIGHTNESS_BANDS, COLLECTION, _dedupe_by_date, to_reflectance, valid_mask
from fetch_sentinel1 import _read_on_grid, open_catalog, run_for_all_mines, select_evenly

BAND_NAMES = ["s2my_bright_early", "s2my_bright_late", "s2my_bright_delta", "s2my_bright_slope",
              "s2my_dark_years_frac", "s2my_dark_onset"]
DARK_THRESHOLD = 0.10   # mittlere Reflektanz (Blau, Grün, Rot, NIR) darunter gilt als "dunkel"


def multiyear_features(year_stack, years, dark_threshold=DARK_THRESHOLD, min_years=4):
    """year_stack: (Jahre, H, W) Helligkeit je Jahr, NaN = kein gültiger Wert.
    years: aufsteigende Jahreszahlen. Rückgabe (6, H, W) in der Reihenfolge von BAND_NAMES."""
    stack = np.asarray(year_stack, dtype=np.float32)
    years = np.asarray(years, dtype=np.float64)
    valid = np.isfinite(stack)
    n_valid = valid.sum(axis=0)
    ok = n_valid >= min_years
    shape = stack.shape[1:]

    def mean_of_first(order):
        total = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.int32)
        for i in order:
            take = valid[i] & (count < 2)
            total[take] += stack[i][take]
            count[take] += 1
        with np.errstate(all="ignore"):
            return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    early = mean_of_first(range(len(years)))
    late = mean_of_first(range(len(years) - 1, -1, -1))

    x = years[:, None, None]
    v = valid.astype(np.float64)
    y = np.where(valid, stack, 0.0).astype(np.float64)
    n = v.sum(axis=0)
    sx, sy = (v * x).sum(axis=0), y.sum(axis=0)
    sxx, sxy = (v * x * x).sum(axis=0), (y * x).sum(axis=0)
    with np.errstate(all="ignore"):
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)

    dark = valid & (np.where(valid, stack, 1.0) < dark_threshold)
    with np.errstate(all="ignore"):
        dark_frac = dark.sum(axis=0) / np.maximum(n_valid, 1)
    onset = np.full(shape, np.nan)
    span = max(years[-1] - years[0], 1.0)
    for i in range(len(years) - 1, -1, -1):           # von hinten, damit das früheste Jahr gewinnt
        onset[dark[i]] = (years[i] - years[0]) / span

    out = np.stack([early, late, late - early, slope, dark_frac, onset]).astype(np.float32)
    out[:, ~ok] = np.nan
    return out


def fetch_for_mine(s2_path, out_path, catalog, years, scenes_per_year, max_cloud, min_years):
    with rasterio.open(s2_path) as ref:
        crs, transform = ref.crs, ref.transform
        width, height = ref.width, ref.height
        bbox = transform_bounds(crs, "EPSG:4326", *ref.bounds)

    def read(it, band, resampling):
        return _read_on_grid(it.assets[band].href, crs, transform, width, height, resampling)

    years = sorted(years)
    stack, with_data = [], []
    for year in years:
        items = [
            it for it in catalog.search(
                collections=[COLLECTION], bbox=bbox, datetime=f"{year}-01-01/{year}-12-31",
            ).items()
            if it.properties.get("eo:cloud_cover", 100) <= max_cloud
        ]
        chosen = select_evenly(_dedupe_by_date(items), scenes_per_year)
        scenes = []
        for it in chosen:
            baseline = it.properties.get("s2:processing_baseline", "04.00")
            bands = [to_reflectance(read(it, b, Resampling.average), baseline) for b in BRIGHTNESS_BANDS]
            with np.errstate(all="ignore"):
                bright = np.mean(bands, axis=0)
            bright[~valid_mask(read(it, "SCL", Resampling.nearest))] = np.nan
            bright[~(bright > 0)] = np.nan
            scenes.append(bright)
        if scenes:
            with np.errstate(all="ignore"):
                stack.append(np.nanmedian(np.stack(scenes), axis=0))
            with_data.append(year)
        else:
            stack.append(np.full((height, width), np.nan, dtype=np.float32))
    if len(with_data) < min_years:
        raise RuntimeError(f"nur {len(with_data)} Jahre mit wolkenarmen Aufnahmen (mind. {min_years})")

    feats = multiyear_features(np.stack(stack), years, min_years=min_years)
    profile = {"driver": "GTiff", "height": height, "width": width, "count": feats.shape[0],
               "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan, "compress": "deflate"}
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(feats)
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(i, name)
    return f"Jahre mit Daten: {with_data[0]}-{with_data[-1]} ({len(with_data)} von {len(years)})"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mehrjahres-Helligkeitsmerkmale (wächst da etwas Dunkles?).")
    parser.add_argument("--imagery-dir", default="data/imagery", metavar="DIR")
    parser.add_argument("--out-dir", default="data/s2_multiyear", metavar="DIR")
    parser.add_argument("--first-year", type=int, default=2018)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--years", default=None,
                        help="Stattdessen eine feste Jahresliste, z.B. 2018,2019,2024,2025 - für die Merkmale "
                             "early/late/delta reichen die ersten und letzten beiden Jahre und der Download "
                             "halbiert sich. Dann --min-years 3 setzen.")
    parser.add_argument("--scenes-per-year", type=int, default=3,
                        help="Aufnahmen pro Jahr, aus denen der Median gebildet wird.")
    parser.add_argument("--max-cloud", type=float, default=15, help="Maximaler Wolkenanteil einer Aufnahme (%%).")
    parser.add_argument("--min-years", type=int, default=4, help="Mindestzahl Jahre mit gültigen Daten pro Pixel.")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    years = ([int(y) for y in args.years.split(",")] if args.years
             else list(range(args.first_year, args.last_year + 1)))
    catalog = open_catalog()
    return run_for_all_mines(
        args.imagery_dir, args.out_dir, args.n_workers, args.overwrite,
        lambda s2, out: fetch_for_mine(s2, out, catalog, years,
                                       args.scenes_per_year, args.max_cloud, args.min_years),
    )


if __name__ == "__main__":
    sys.exit(main())
