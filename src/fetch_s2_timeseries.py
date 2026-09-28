"""
Sentinel-2-Zeitreihen-Merkmale pro Mine
========================================

Idee: Schatten wandern mit dem Sonnenstand, Reifen nicht. In Antofagasta
steht die Sonne beim Sentinel-2-Überflug im Dezember ~23° vom Zenit, im Juni
~56°. Ein Berghang oder eine Grubenkante liegt deshalb im Winter im
Schatten und im Sommer in der Sonne - die Helligkeit schwankt stark. Eine
Reifenhalde ist das ganze Jahr ungefähr gleich dunkel.

Für jedes Minenbild werden bis zu max_scenes wolkenarme Sentinel-2-
Aufnahmen (L2A, Planetary Computer, kostenlos, ohne Account) gleichmäßig
übers Jahr verteilt geladen, direkt aufs Pixelraster des Minenbilds
gebracht, Wolken und Wolkenschatten per Szenenklassifikation (SCL)
ausmaskiert, und pro Pixel die Helligkeit (Mittel aus Blau, Grün, Rot,
NIR) über alle Aufnahmen ausgewertet:

  - s2t_bright_cv: Standardabweichung / Mittelwert (0 = immer gleich hell)
  - s2t_bright_min_ratio: dunkelste Aufnahme / Median (1 = nie dunkler als
    normal; klein = zeitweise stark verschattet)

Pixel mit weniger als min_valid gültigen Aufnahmen bleiben leer (NaN).

Nutzung:
    python src/fetch_s2_timeseries.py --imagery-dir data/imagery --out-dir data/s2_temporal
"""

import argparse
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from fetch_sentinel1 import _read_on_grid, open_catalog, run_for_all_mines, select_evenly

COLLECTION = "sentinel-2-l2a"
BRIGHTNESS_BANDS = ("B02", "B03", "B04", "B08")
# SCL-Klassen, die verworfen werden: 0 keine Daten, 1 gesättigt/defekt,
# 3 Wolkenschatten, 8/9 Wolken, 10 Cirrus. Klasse 2 ("dunkle Bereiche",
# u.a. Geländeschatten) bleibt ausdrücklich DRIN - genau die soll ja erkannt
# werden.
SCL_INVALID = (0, 1, 3, 8, 9, 10)


def to_reflectance(dn, processing_baseline):
    """Rohwerte (DN) -> Reflektanz 0..1. Seit Verarbeitungsversion 04.00
    (2022) sind die L2A-Werte um 1000 verschoben. DN 0 = keine Daten."""
    dn = dn.astype(np.float32)
    offset = 1000.0 if float(processing_baseline) >= 4.0 else 0.0
    refl = (dn - offset) / 10000.0
    refl[dn == 0] = np.nan
    return refl


def valid_mask(scl):
    """True, wo die Szenenklassifikation weder Wolke noch Wolkenschatten
    noch fehlende Daten meldet."""
    return np.isfinite(scl) & ~np.isin(scl, SCL_INVALID)


def temporal_stats(brightness, min_valid=3):
    """brightness: (Aufnahmen, H, W) mit NaN für ungültige Werte.
    Rückgabe (2, H, W): Variationskoeffizient und Minimum/Median."""
    stack = brightness.astype(np.float32)
    stack[~(stack > 0)] = np.nan  # Reflektanz <= 0 ist physikalisch unsinnig
    n_valid = np.isfinite(stack).sum(axis=0)
    enough = n_valid >= min_valid

    cv = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    min_ratio = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    if enough.any():
        s = stack[:, enough]
        with np.errstate(all="ignore"):
            cv[enough] = np.nanstd(s, axis=0) / np.nanmean(s, axis=0)
            min_ratio[enough] = np.nanmin(s, axis=0) / np.nanmedian(s, axis=0)
    return np.stack([cv, min_ratio])


def _dedupe_by_date(items):
    """Dieselbe Aufnahme taucht an Kachelgrenzen mehrfach auf (eine pro
    MGRS-Kachel) -> pro Tag nur eine behalten."""
    seen, out = set(), []
    for it in sorted(items, key=lambda i: i.datetime):
        day = it.datetime.date()
        if day not in seen:
            seen.add(day)
            out.append(it)
    return out


def fetch_for_mine(s2_path, out_path, catalog, start, end, max_scenes, max_cloud, min_valid):
    with rasterio.open(s2_path) as ref:
        crs, transform = ref.crs, ref.transform
        width, height = ref.width, ref.height
        bbox = transform_bounds(crs, "EPSG:4326", *ref.bounds)

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

    brightness = []
    for it in chosen:
        baseline = it.properties.get("s2:processing_baseline", "04.00")
        bands = [to_reflectance(read(it, b, Resampling.average), baseline) for b in BRIGHTNESS_BANDS]
        with np.errstate(all="ignore"):
            bright = np.mean(bands, axis=0)
        bright[~valid_mask(read(it, "SCL", Resampling.nearest))] = np.nan
        brightness.append(bright)

    stats = temporal_stats(np.stack(brightness), min_valid=min_valid)

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": stats.shape[0],
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stats)
        dst.set_band_description(1, "s2t_bright_cv")
        dst.set_band_description(2, "s2t_bright_min_ratio")
    months = sorted({it.datetime.month for it in chosen})
    return f"{len(chosen)} Aufnahmen (Monate {months})"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Berechnet Helligkeits-Zeitreihen-Merkmale aus mehreren "
                    "Sentinel-2-Aufnahmen pro Mine (Schatten vs. Reifen)."
    )
    parser.add_argument("--imagery-dir", default="data/imagery", metavar="DIR",
                        help="Ordner mit den Sentinel-2-GeoTIFFs (ein Bild pro Mine).")
    parser.add_argument("--out-dir", default="data/s2_temporal", metavar="DIR",
                        help="Zielordner für die Zeitreihen-GeoTIFFs.")
    parser.add_argument("--start", default="2023-01-01",
                        help="Beginn des Zeitraums (JJJJ-MM-TT). Sollte ein ganzes Jahr "
                             "abdecken, damit Sommer- und Wintersonnenstand dabei sind.")
    parser.add_argument("--end", default="2023-12-31", help="Ende des Zeitraums (JJJJ-MM-TT).")
    parser.add_argument("--max-scenes", type=int, default=8,
                        help="Maximale Anzahl Aufnahmen pro Mine (gleichmäßig übers Jahr).")
    parser.add_argument("--max-cloud", type=float, default=20,
                        help="Maximaler Wolkenanteil einer Aufnahme in Prozent.")
    parser.add_argument("--min-valid", type=int, default=3,
                        help="Mindestanzahl gültiger Aufnahmen pro Pixel.")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Anzahl Minen, die gleichzeitig heruntergeladen werden.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bereits vorhandene Dateien neu erzeugen.")
    args = parser.parse_args(argv)

    catalog = open_catalog()
    return run_for_all_mines(
        args.imagery_dir, args.out_dir, args.n_workers, args.overwrite,
        lambda s2, out: fetch_for_mine(s2, out, catalog, args.start, args.end,
                                       args.max_scenes, args.max_cloud, args.min_valid),
    )


if __name__ == "__main__":
    sys.exit(main())
