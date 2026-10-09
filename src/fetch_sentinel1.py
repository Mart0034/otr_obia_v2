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
import time
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


def _read_on_grid_once(href, dst_crs, dst_transform, width, height, resampling):
    with rasterio.open(href) as src:
        with WarpedVRT(
            src, crs=dst_crs, transform=dst_transform, width=width, height=height,
            resampling=resampling, src_nodata=src.nodata, nodata=np.nan,
            dtype="float32",
        ) as vrt:
            return vrt.read(1)


def _read_on_grid(href, dst_crs, dst_transform, width, height, resampling=Resampling.average,
                  retries=3, retry_delay=2.0):
    """Liest nur den benötigten Ausschnitt einer (Cloud-Optimized-)Aufnahme
    und bringt ihn direkt aufs Zielraster. Kurze Netzwerkaussetzer (OSError,
    dazu gehört auch rasterios RasterioIOError) werden mit wachsender Pause
    wiederholt - sonst verliert ein einziger Aussetzer die ganze Mine."""
    for attempt in range(retries):
        try:
            return _read_on_grid_once(href, dst_crs, dst_transform, width, height, resampling)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))


def parallel_map(fn, items, n_threads):
    """fn auf alle items anwenden, mit bis zu n_threads Threads (Download ist
    I/O-gebunden); Ergebnisreihenfolge = Eingabereihenfolge, Fehler werden
    weitergereicht."""
    items = list(items)
    if n_threads <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(n_threads, len(items))) as pool:
        return list(pool.map(fn, items))


def fetch_for_mine(s2_path, out_path, catalog, start, end, max_scenes, read_threads=3):
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

    jobs = [(band, it) for band in S1_BANDS for it in chosen]
    arrays = parallel_map(
        lambda j: _read_on_grid(j[1].assets[j[0]].href, crs, transform, width, height),
        jobs, read_threads)
    bands = [combine_scenes(arrays[i * len(chosen):(i + 1) * len(chosen)]) for i in range(len(S1_BANDS))]

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
    parser.add_argument("--read-threads", type=int, default=3,
                        help="Parallele Lesezugriffe innerhalb einer Mine (Szenen/Bänder).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bereits vorhandene Dateien neu erzeugen.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Auch Minen erneut versuchen, die beim letzten Mal endgültig "
                             "fehlschlugen (z.B. keine Aufnahmen vorhanden).")
    args = parser.parse_args(argv)

    catalog = open_catalog()
    return run_for_all_mines(
        args.imagery_dir, args.out_dir, args.n_workers, args.overwrite,
        lambda s2, out: fetch_for_mine(s2, out, catalog, args.start, args.end, args.max_scenes,
                                       args.read_threads),
        retry_failed=args.retry_failed,
    )


def _tune_gdal_http():
    """GDAL-Netzwerkeinstellungen für viele kleine Bereichsabfragen auf Cloud-
    Optimized-GeoTIFFs (nur Defaults - bereits gesetzte Umgebungsvariablen
    gewinnen): gebündelte/parallele HTTP-Anfragen, kein Verzeichnis-Listing
    beim Öffnen, Wiederholungen bei Aussetzern."""
    for key, value in (
        ("GDAL_HTTP_MULTIPLEX", "YES"),
        ("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES"),
        ("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"),
        ("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff"),
        ("GDAL_HTTP_MAX_RETRY", "3"),
        ("GDAL_HTTP_RETRY_DELAY", "2"),
        ("VSI_CACHE", "TRUE"),
    ):
        os.environ.setdefault(key, value)


def open_catalog():
    """Logging einrichten, Windows-Pfad-Workaround anwenden und den
    Planetary-Computer-Katalog öffnen (anonym, ohne Account)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    _ascii_safe_gdal_paths()
    _tune_gdal_http()

    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _fail_marker(out):
    return out + ".failed"


def _tile_done(out, overwrite, retry_failed):
    """True, wenn die Mine nicht erneut geladen werden muss: Ausgabe vorhanden
    oder (ohne retry_failed) beim letzten Mal endgültig fehlgeschlagen."""
    if overwrite:
        return False
    if os.path.exists(out):
        return True
    return os.path.exists(_fail_marker(out)) and not retry_failed


def _fmt_dur(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def fetch_one(path, out, fetch_fn):
    """Eine Mine laden - atomar: erst in <out>.part schreiben, dann umbenennen,
    damit ein abgebrochener Lauf nie eine halbe Datei hinterlässt, die beim
    nächsten Mal als 'fertig' gilt. RuntimeError (= die Daten gibt es nicht,
    z.B. keine Aufnahmen) wird als endgültig gemerkt (<out>.failed); andere
    Fehler (Netzwerk, Speicher) gelten als vorübergehend und werden beim
    nächsten Lauf erneut versucht. Gibt (Meldung, Dauer in s) zurück."""
    t0 = time.time()
    part = out + ".part"
    try:
        if os.path.exists(_fail_marker(out)):
            os.remove(_fail_marker(out))
        msg = fetch_fn(path, part)
        os.replace(part, out)
        return msg, time.time() - t0
    except BaseException as e:
        if os.path.exists(part):
            os.remove(part)
        if isinstance(e, RuntimeError):
            with open(_fail_marker(out), "w") as f:
                f.write(str(e))
        raise


def run_for_all_mines(imagery_dir, out_dir, n_workers, overwrite, fetch_fn, retry_failed=False):
    """Ruft fetch_fn(s2_pfad, ausgabe_pfad) für jedes Minenbild auf, parallel
    in Threads. Bereits vorhandene Ausgaben werden übersprungen (außer bei
    overwrite), ebenso Minen, die endgültig fehlschlugen (außer bei
    retry_failed); Fehler einzelner Minen stoppen die übrigen nicht. Ausgaben
    werden atomar geschrieben (siehe fetch_one), das Log zeigt Dauer pro Mine
    und eine Restzeit-Schätzung. fetch_fn gibt einen kurzen Text für das Log zurück."""
    os.makedirs(out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, "*.part")):
        os.remove(stale)   # Reste eines abgebrochenen Laufs
    paths = sorted(glob.glob(os.path.join(imagery_dir, "*.tif")))
    todo = []
    skipped_failed = 0
    for p in paths:
        out = os.path.join(out_dir, os.path.basename(p))
        if _tile_done(out, overwrite, retry_failed):
            skipped_failed += (not os.path.exists(out))
            continue
        todo.append((p, out))
    logger.info("%d Minenbilder gefunden, %d müssen noch geladen werden%s.", len(paths), len(todo),
                f" ({skipped_failed} übersprungen, weil dort keine Daten vorhanden sind - "
                f"--retry-failed zum erneuten Versuch)" if skipped_failed else "")

    failed = []
    t_start, done, busy = time.time(), 0, 0.0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fetch_one, p, out, fetch_fn): p for p, out in todo}
        for fut in as_completed(futures):
            name = os.path.basename(futures[fut])
            done += 1
            try:
                msg, dur = fut.result()
                busy += dur
                elapsed = time.time() - t_start
                eta = elapsed / done * (len(todo) - done)
                logger.info("  [%d/%d] %s: %s (%.0f s, noch ca. %s)", done, len(todo), name, msg, dur,
                            _fmt_dur(eta))
            except Exception as e:
                logger.error("  [%d/%d] %s: fehlgeschlagen (%s)", done, len(todo), name, e)
                failed.append(name)
    if todo:
        wall = time.time() - t_start
        logger.info("Fertig in %s: %d Minen, im Schnitt %.0f s pro Mine (%d parallel; jede Mine einzeln "
                    "im Schnitt %.0f s - ist das deutlich mehr als bei n_workers=1, bremst der Server).",
                    _fmt_dur(wall), done, wall / max(done, 1), n_workers, busy / max(done, 1))

    if failed:
        logger.warning("%d Minen fehlgeschlagen: %s", len(failed), failed)
    return 1 if failed and len(failed) == len(todo) else 0


if __name__ == "__main__":
    sys.exit(main())
