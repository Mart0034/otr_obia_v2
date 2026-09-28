"""
Sentinel-1-Zeitreihen-Merkmale pro Mine
=========================================

Idee: Radar-Rückstreuung hängt stark vom Blickwinkel ab (Einfallswinkel,
Aufnahmerichtung aufsteigend/absteigend). Eine feste, geometrisch klare
Fläche - eine Fahrstraße, eine Gebäudekante, eine Grubenwand - streut je
nach Blickrichtung unterschiedlich stark zurück, ihre Rückstreuung
schwankt also zwischen einzelnen Aufnahmen. Ein ungeordneter Reifenhaufen
streut dagegen diffus in praktisch alle Richtungen und bleibt über
verschiedene Aufnahmen hinweg vergleichsweise konstant. Dieselbe Logik wie
bei den Sentinel-2-Zeitreihen-Merkmalen (Schatten wandern mit dem
Sonnenstand, Reifen nicht), nur für Radar statt Blickwinkel statt Sonne.

Für jedes Minenbild werden bis zu max_scenes Sentinel-1-Aufnahmen (RTC,
Planetary Computer, kostenlos, ohne Account) im Zeitraum geladen, aufs
Pixelraster des Minenbilds gebracht, und pro Pixel der Variationskoeffizient
(Standardabweichung / Mittelwert) der linearen Rückstreuung über alle
Aufnahmen berechnet:

  - s1t_vv_cv, s1t_vh_cv: 0 = über alle Aufnahmen exakt gleiche
    Rückstreuung, hoch = starke blickwinkelabhängige Schwankung

Ergänzt die bereits vorhandenen s1_vv_std/s1_vh_std (räumliche Textur
INNERHALB einer Aufnahme, siehe fetch_sentinel1.py) um eine zeitliche
Variante.

Nutzung:
    python src/fetch_sentinel1_timeseries.py --imagery-dir data/imagery --out-dir data/s1_temporal

Ausgabe: je Mine ein GeoTIFF mit 2 Bändern (s1t_vv_cv, s1t_vh_cv).
"""

import argparse
import sys

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from fetch_sentinel1 import S1_BANDS, _read_on_grid, open_catalog, run_for_all_mines, select_evenly


def temporal_cv(stack, min_valid=4):
    """stack: (Aufnahmen, H, W) lineare Rückstreuung, <=0/NaN = ungültig.
    Rückgabe (H, W): Standardabweichung / Mittelwert über die gültigen
    Aufnahmen je Pixel. Pixel mit weniger als min_valid gültigen Aufnahmen
    bleiben leer (NaN)."""
    stack = stack.astype(np.float32).copy()
    stack[~(stack > 0)] = np.nan
    n_valid = np.isfinite(stack).sum(axis=0)
    enough = n_valid >= min_valid

    cv = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    if enough.any():
        s = stack[:, enough]
        with np.errstate(all="ignore"):
            cv[enough] = np.nanstd(s, axis=0) / np.nanmean(s, axis=0)
    return cv


def fetch_for_mine(s2_path, out_path, catalog, start, end, max_scenes, min_valid):
    with rasterio.open(s2_path) as ref:
        crs, transform = ref.crs, ref.transform
        width, height = ref.width, ref.height
        bbox = transform_bounds(crs, "EPSG:4326", *ref.bounds)

    items = list(catalog.search(
        collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start}/{end}",
    ).items())
    chosen = select_evenly(items, max_scenes)
    if len(chosen) < min_valid:
        raise RuntimeError(
            f"nur {len(chosen)} Sentinel-1-Aufnahmen zwischen {start} und {end} "
            f"gefunden (mind. {min_valid} nötig)"
        )

    cv_bands = []
    for band in S1_BANDS:
        scenes = np.stack([
            _read_on_grid(it.assets[band].href, crs, transform, width, height)
            for it in chosen
        ])
        cv_bands.append(temporal_cv(scenes, min_valid=min_valid))

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": len(S1_BANDS),
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.stack(cv_bands))
        for i, band in enumerate(S1_BANDS, start=1):
            dst.set_band_description(i, f"s1t_{band}_cv")
    return f"{len(chosen)} Aufnahmen"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Berechnet Rückstreu-Zeitreihen-Merkmale (Blickwinkel-"
                    "Variabilität) aus mehreren Sentinel-1-Aufnahmen pro Mine."
    )
    parser.add_argument("--imagery-dir", default="data/imagery", metavar="DIR",
                        help="Ordner mit den Sentinel-2-GeoTIFFs (ein Bild pro Mine).")
    parser.add_argument("--out-dir", default="data/s1_temporal", metavar="DIR",
                        help="Zielordner für die Zeitreihen-GeoTIFFs.")
    parser.add_argument("--start", default="2023-01-01",
                        help="Beginn des Zeitraums (JJJJ-MM-TT). Sollte ein ganzes Jahr "
                             "abdecken, damit unterschiedliche Bahnrichtungen dabei sind.")
    parser.add_argument("--end", default="2023-12-31", help="Ende des Zeitraums (JJJJ-MM-TT).")
    parser.add_argument("--max-scenes", type=int, default=12,
                        help="Maximale Anzahl Aufnahmen pro Mine (gleichmäßig übers Jahr).")
    parser.add_argument("--min-valid", type=int, default=4,
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
                                       args.max_scenes, args.min_valid),
    )


if __name__ == "__main__":
    sys.exit(main())
