"""
Sentinel-1-Radardaten pro Mine laden
=====================================

Holt für jedes Sentinel-2-Minenbild in imagery_dir die passenden
Sentinel-1-Radaraufnahmen (VV/VH, radiometrisch geländekorrigiert, "RTC")
von Microsoft Planetary Computer - kostenlos, ohne Account/Login.

Pro Mine werden bis zu max_scenes Aufnahmen aus dem gewählten Zeitraum
gleichmäßig verteilt ausgewählt und pixelweise per Median kombiniert
(unterdrückt das für Radar typische "Speckle"-Rauschen). Das Ergebnis wird
auf EXAKT dasselbe Pixelraster wie das Sentinel-2-Bild der Mine gebracht
(gleiches Koordinatensystem, gleiche Pixel) und als GeoTIFF mit gleichem
Dateinamen in out_dir gespeichert. Die Pipeline (--s1-dir) kann die Werte
dadurch direkt pro Segment auswerten.

Warum Radar: Radar misst die Oberflächenrauheit statt der Farbe. Ein
Reifenhaufen streut das Signal stark, eine planierte Fahrstraße kaum -
genau die Verwechslung, die im optischen Sentinel-2-Bild die meisten
Falsch-Positiven erzeugt. Außerdem wolkenunabhängig.

Nutzung:
    python src/fetch_sentinel1.py --imagery-dir data/imagery --out-dir data/sentinel1

Ausgabe: je Mine ein GeoTIFF mit 2 Bändern (s1_vv_db, s1_vh_db), Werte in dB.
"""

import argparse
import glob
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
S1_BANDS = ("vv", "vh")


def select_evenly(items, max_scenes):
    """Wählt bis zu max_scenes Aufnahmen gleichmäßig über den Zeitraum verteilt
    aus (statt z.B. nur die ersten N), damit saisonale Schwankungen im Median
    ausgeglichen werden."""
    items = sorted(items, key=lambda it: it.datetime)
    if len(items) <= max_scenes:
        return items
    idx = np.unique(np.linspace(0, len(items) - 1, max_scenes).round().astype(int))
    return [items[i] for i in idx]


def combine_scenes(scenes):
    """Kombiniert mehrere linear skalierte Rückstreu-Arrays (gleiche Form)
    per pixelweisem Median und rechnet in dB um. Werte <= 0 bzw. NaN gelten
    als "keine Daten" (Rand der Aufnahme). Pixel ohne jede gültige Aufnahme
    bleiben NaN."""
    stack = np.stack(scenes).astype(np.float32)
    stack[~(stack > 0)] = np.nan
    with np.errstate(all="ignore"):
        valid = np.isfinite(stack).any(axis=0)
        median = np.full(stack.shape[1:], np.nan, dtype=np.float32)
        median[valid] = np.nanmedian(stack[:, valid], axis=0)
        return 10 * np.log10(median)


def _read_on_grid(href, dst_crs, dst_transform, width, height, resampling=Resampling.average):
    """Liest nur den benötigten Ausschnitt einer (Cloud-Optimized-)Aufnahme
    und bringt ihn direkt aufs Zielraster."""
    with rasterio.open(href) as src:
        with WarpedVRT(
            src, crs=dst_crs, transform=dst_transform, width=width, height=height,
            resampling=resampling, src_nodata=src.nodata, nodata=np.nan,
            dtype="float32",
        ) as vrt:
            return vrt.read(1)


def fetch_for_mine(s2_path, out_path, catalog, start, end, max_scenes):
    with rasterio.open(s2_path) as ref:
        crs, transform = ref.crs, ref.transform
        width, height = ref.width, ref.height
        bbox = transform_bounds(crs, "EPSG:4326", *ref.bounds)

    items = list(catalog.search(
        collections=[COLLECTION], bbox=bbox, datetime=f"{start}/{end}",
    ).items())
    if not items:
        raise RuntimeError(f"keine Sentinel-1-Aufnahmen zwischen {start} und {end} gefunden")
    chosen = select_evenly(items, max_scenes)

    bands = []
    for band in S1_BANDS:
        scenes = [
            _read_on_grid(it.assets[band].href, crs, transform, width, height)
            for it in chosen
        ]
        bands.append(combine_scenes(scenes))

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": len(S1_BANDS),
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.stack(bands))
        for i, band in enumerate(S1_BANDS, start=1):
            dst.set_band_description(i, f"s1_{band}_db")
    return f"{len(chosen)} Radaraufnahmen kombiniert"


def _ascii_safe_gdal_paths():
    """Unter Windows kann der Download-Teil von GDAL (curl/schannel) die
    Zertifikatsdatei nicht öffnen, wenn ihr Pfad Nicht-ASCII-Zeichen enthält
    (z.B. 'Martín' im Benutzernamen) - jeder Download schlägt dann fehl.
    rasterio setzt beim Import GDAL_CURL_CA_BUNDLE auf certifi; hier werden
    diese Pfade durch ihre reine ASCII-8.3-Kurzform ersetzt. GDAL liest die
    Variablen erst beim Zugriff, daher reicht es, das nach dem Import zu tun."""
    if sys.platform != "win32":
        return
    import ctypes

    import certifi

    def short(path):
        buf = ctypes.create_unicode_buffer(1024)
        ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024)
        return buf.value or path

    for var in ("GDAL_CURL_CA_BUNDLE", "PROJ_CURL_CA_BUNDLE", "CURL_CA_BUNDLE"):
        value = os.environ.get(var) or certifi.where()
        if not value.isascii():
            value = short(value)
        os.environ[var] = value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lädt Sentinel-1-Radardaten passend zu jedem Sentinel-2-Minenbild."
    )
    parser.add_argument("--imagery-dir", default="data/imagery", metavar="DIR",
                        help="Ordner mit den Sentinel-2-GeoTIFFs (ein Bild pro Mine).")
    parser.add_argument("--out-dir", default="data/sentinel1", metavar="DIR",
                        help="Zielordner für die Sentinel-1-GeoTIFFs.")
    parser.add_argument("--start", default="2023-01-01",
                        help="Beginn des Zeitraums (JJJJ-MM-TT). Idealerweise derselbe "
                             "Zeitraum wie das Sentinel-2-Komposit.")
    parser.add_argument("--end", default="2023-12-31", help="Ende des Zeitraums (JJJJ-MM-TT).")
    parser.add_argument("--max-scenes", type=int, default=12,
                        help="Maximale Anzahl Aufnahmen pro Mine für den Median.")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Anzahl Minen, die gleichzeitig heruntergeladen werden.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bereits vorhandene Dateien neu erzeugen.")
    args = parser.parse_args(argv)

    catalog = open_catalog()
    return run_for_all_mines(
        args.imagery_dir, args.out_dir, args.n_workers, args.overwrite,
        lambda s2, out: fetch_for_mine(s2, out, catalog, args.start, args.end, args.max_scenes),
    )


def open_catalog():
    """Logging einrichten, Windows-Pfad-Workaround anwenden und den
    Planetary-Computer-Katalog öffnen (anonym, ohne Account)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    _ascii_safe_gdal_paths()

    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def run_for_all_mines(imagery_dir, out_dir, n_workers, overwrite, fetch_fn):
    """Ruft fetch_fn(s2_pfad, ausgabe_pfad) für jedes Minenbild auf, parallel
    in Threads. Bereits vorhandene Ausgaben werden übersprungen (außer bei
    overwrite), Fehler einzelner Minen stoppen die übrigen nicht. fetch_fn
    gibt einen kurzen Text für das Log zurück."""
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(imagery_dir, "*.tif")))
    todo = []
    for p in paths:
        out = os.path.join(out_dir, os.path.basename(p))
        if os.path.exists(out) and not overwrite:
            continue
        todo.append((p, out))
    logger.info("%d Minenbilder gefunden, %d müssen noch geladen werden.", len(paths), len(todo))

    failed = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fetch_fn, p, out): p for p, out in todo}
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
