"""
Laden und Verarbeiten gleichzeitig: jede Mine wird verarbeitet, sobald ihre Daten da sind
=========================================================================================

Bisher: erst ALLE Minen herunterladen (S1, DEM, Mehrjahres-Helligkeit), dann erst
Segmentierung + Merkmale. Hier laufen beide Schritte ineinander: Download-Threads holen
die Daten einer Mine, und sobald sie komplett sind, wird die Mine sofort in einem eigenen
Prozess segmentiert und mit Merkmalen versehen, während schon die nächsten Minen laden.
Die Gesamtzeit ist dann etwa die des langsameren Schritts statt der Summe.

Das Ergebnis ist derselbe Zwischenspeicher wie ihn die Pipeline selbst schreibt
(feature_cache.pkl + polygons_cache.gpkg im --output-dir). Training/Export folgen
danach wie gewohnt mit --use-cache - oder gleich hier mit --train.

Die Mine-Bilder (--imagery-dir) müssen schon vorhanden sein; geladen werden die
Zusatzdaten, deren Ordner angegeben sind (--s1-dir, --dem-dir, --s2my-dir). Bereits
geladene Dateien werden übersprungen; mit --resume gilt das auch für Minen, die schon im
Zwischenspeicher stehen (nur sinnvoll, wenn sich an den Merkmalen nichts geändert hat).

Beispiel:
    python src/stream_pipeline.py --imagery-dir data/imagery --labels-path data/dump_labels.gpkg \\
        --s1-dir data/sentinel1 --dem-dir data/dem --s2my-dir data/s2_multiyear \\
        --s2my-bands s2my_bright_early,s2my_bright_late --target-segment-px 12 \\
        --output-dir out_stream --fetch-workers 4 --n-jobs 8 --train
"""

import glob
import logging
import os
import shutil
import sys
import tempfile
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


def run_stream(tiles, fetch_fn, process_fn, fetch_workers, executor_factory):
    """tiles: Pfade. fetch_fn(tile) lädt alles Nötige (wirft bei Fehlschlag),
    process_fn(tile) -> Ergebnis wird in einem Executor aus executor_factory(n)
    ausgeführt, sobald fetch_fn für die Mine fertig ist. Gibt (Ergebnisse je Mine,
    Liste (Mine, Phase, Fehler)) zurück. Abgestürzte Prozess-Pools (meist
    Speichermangel) werden einzeln und isoliert wiederholt."""
    results, failed, crashed = {}, [], []
    n_total, counters = len(tiles), {"fetched": 0, "processed": 0}
    t0 = time.time()

    def _on_processed(tile):
        def cb(fut):
            counters["processed"] += 1
            tag = "ok" if not fut.exception() else "FEHLER"
            logger.info("  verarbeitet [%d/%d] %s (%s, %.0f s seit Start)", counters["processed"], n_total,
                        os.path.basename(tile), tag, time.time() - t0)
        return cb

    pfut = {}
    executor = executor_factory(None)
    try:
        with ThreadPoolExecutor(max_workers=fetch_workers) as fpool:
            ffut = {fpool.submit(fetch_fn, t): t for t in tiles}
            for f in as_completed(ffut):
                t = ffut[f]
                counters["fetched"] += 1
                try:
                    f.result()
                except Exception as e:
                    logger.error("  geladen [%d/%d] %s: fehlgeschlagen (%s) - Mine wird übersprungen",
                                 counters["fetched"], n_total, os.path.basename(t), e)
                    failed.append((t, "download", str(e)))
                    continue
                logger.info("  geladen [%d/%d] %s -> Verarbeitung gestartet", counters["fetched"], n_total,
                            os.path.basename(t))
                pf = executor.submit(process_fn, t)
                pf.add_done_callback(_on_processed(t))
                pfut[pf] = t
        for pf in as_completed(pfut):
            t = pfut[pf]
            try:
                results[t] = pf.result()
            except BrokenProcessPool:
                crashed.append(t)
            except Exception as e:
                logger.error("  %s: Verarbeitung fehlgeschlagen (%s)", os.path.basename(t), e)
                failed.append((t, "processing", str(e)))
    finally:
        executor.shutdown(wait=True)

    for t in sorted(crashed, key=lambda p: -os.path.getsize(p)):
        logger.warning("Worker abgestürzt (vermutlich Speichermangel) - %s wird einzeln wiederholt.",
                       os.path.basename(t))
        solo = executor_factory(1)
        try:
            results[t] = solo.submit(process_fn, t).result()
        except Exception as e:
            logger.error("  %s: auch einzeln fehlgeschlagen (%s)", os.path.basename(t), e)
            failed.append((t, "processing", str(e)))
        finally:
            solo.shutdown(wait=True)
    return results, failed


def process_tile(path, cfg):
    """Eine Mine wie build_dataset() verarbeiten (Top-Level-Funktion für den Prozess-Pool).
    build_dataset arbeitet auf einem Ordner; dafür ein temporärer Ordner mit einem Link."""
    from otr_obia_pipeline import build_dataset

    tmp = tempfile.mkdtemp(prefix="stream_tile_")
    try:
        os.symlink(os.path.abspath(path), os.path.join(tmp, os.path.basename(path)))
        return build_dataset({**cfg, "imagery_dir": tmp, "n_jobs": 1})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def combine_results(results, order):
    """results: {tile: (feature_df, polygons_gdf)} -> ein Datensatz in der Reihenfolge von
    order, Polygone im Koordinatensystem der ersten Mine."""
    parts = [results[t] for t in order if t in results]
    feats = pd.concat([f for f, _ in parts], ignore_index=True)
    crs = parts[0][1].crs
    polys = pd.concat([p.to_crs(crs) if p.crs != crs else p for _, p in parts], ignore_index=True)
    return feats, gpd.GeoDataFrame(polys, crs=crs)


def write_cache(feature_df, polygons_gdf, output_dir):
    """Atomar schreiben (erst .tmp, dann umbenennen) - ein Abbruch hinterlässt keinen halben Cache."""
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, "feature_cache.pkl")
    ppath = os.path.join(output_dir, "polygons_cache.gpkg")
    feature_df.to_pickle(fpath + ".tmp")
    os.replace(fpath + ".tmp", fpath)
    polygons_gdf.to_file(ppath + ".tmp.gpkg", driver="GPKG")
    os.replace(ppath + ".tmp.gpkg", ppath)
    return fpath, ppath


def make_fetchers(cfg, args, catalog):
    """Liste (Name, Zielordner, fetch_fn(s2_pfad, ausgabe_pfad)) der Daten, die zu laden sind."""
    out = []
    if cfg.get("s1_dir"):
        from fetch_sentinel1 import fetch_for_mine as fetch_s1
        out.append(("s1", cfg["s1_dir"], lambda s2, o: fetch_s1(
            s2, o, catalog, args.s1_start, args.s1_end, args.s1_max_scenes, args.read_threads)))
    if cfg.get("dem_dir"):
        from fetch_dem import fetch_for_mine as fetch_dem
        out.append(("dem", cfg["dem_dir"], lambda s2, o: fetch_dem(s2, o, catalog)))
    if cfg.get("s2my_dir"):
        from fetch_s2_multiyear import fetch_for_mine as fetch_my
        years = [int(y) for y in args.years.split(",")]
        out.append(("s2my", cfg["s2my_dir"], lambda s2, o: fetch_my(
            s2, o, catalog, years, args.scenes_per_year, args.max_cloud, args.min_years, args.read_threads)))
    return out


def fetch_all_for_tile(path, fetchers, retry_failed):
    """Alle fehlenden Zusatzdaten einer Mine laden (atomar, mit Fehler-Markern wie die
    einzelnen fetch_*-Skripte). Wirft, wenn etwas fehlt."""
    from fetch_sentinel1 import _fail_marker, fetch_one

    for name, out_dir, fn in fetchers:
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(out):
            continue
        if os.path.exists(_fail_marker(out)) and not retry_failed:
            raise RuntimeError(f"{name}: dort gibt es keine Daten (--retry-failed für neuen Versuch)")
        fetch_one(path, out, fn)


def main(argv=None):
    from cli import build_arg_parser, resolve_config
    from otr_obia_pipeline import CONFIG, _init_worker_logging, extract_mine_id
    from fetch_sentinel1 import open_catalog

    parser = build_arg_parser("Laden und Verarbeiten gleichzeitig (siehe Modul-Docstring).")
    parser.add_argument("--fetch-workers", type=int, default=4, help="Minen, die gleichzeitig geladen werden.")
    parser.add_argument("--read-threads", type=int, default=3, help="Parallele Lesezugriffe je Mine.")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Minen überspringen, die schon im Zwischenspeicher im --output-dir stehen.")
    parser.add_argument("--train", action="store_true", help="Danach Training + Export (wie --use-cache).")
    parser.add_argument("--s1-start", default="2023-01-01")
    parser.add_argument("--s1-end", default="2023-12-31")
    parser.add_argument("--s1-max-scenes", type=int, default=12)
    parser.add_argument("--years", default="2018,2019,2024,2025")
    parser.add_argument("--scenes-per-year", type=int, default=3)
    parser.add_argument("--max-cloud", type=float, default=15)
    parser.add_argument("--min-years", type=int, default=3)
    args = parser.parse_args(argv)
    cfg = resolve_config(CONFIG, args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    tiles = sorted(glob.glob(os.path.join(cfg["imagery_dir"], "*.tif")), key=os.path.getsize, reverse=True)
    if not tiles:
        raise FileNotFoundError(f"Keine GeoTIFFs in {cfg['imagery_dir']}")

    old = None
    cache_f = os.path.join(cfg["output_dir"], "feature_cache.pkl")
    cache_p = os.path.join(cfg["output_dir"], "polygons_cache.gpkg")
    if args.resume and os.path.exists(cache_f) and os.path.exists(cache_p):
        old = (pd.read_pickle(cache_f), gpd.read_file(cache_p))
        done_ids = set(old[0]["mine_id"].astype(str))
        before = len(tiles)
        tiles = [t for t in tiles if str(extract_mine_id(t)) not in done_ids]
        logger.info("--resume: %d von %d Minen stehen schon im Zwischenspeicher.", before - len(tiles), before)

    fetchers = make_fetchers(cfg, args, open_catalog()) if any(cfg.get(k) for k in ("s1_dir", "dem_dir", "s2my_dir")) else []
    logger.info("%d Minen; Zusatzdaten werden geladen für: %s", len(tiles), [n for n, _, _ in fetchers] or "nichts")

    n_proc = max(1, min(cfg.get("n_jobs") or os.cpu_count() or 1, len(tiles)))

    def factory(n):
        return ProcessPoolExecutor(max_workers=n or n_proc, initializer=_init_worker_logging)

    t0 = time.time()
    results, failed = run_stream(
        tiles, lambda t: fetch_all_for_tile(t, fetchers, args.retry_failed),
        partial(process_tile, cfg=cfg), args.fetch_workers, factory)
    logger.info("Laden + Verarbeiten fertig in %.1f min: %d Minen ok, %d fehlgeschlagen.",
                (time.time() - t0) / 60, len(results), len(failed))
    for t, phase, err in failed:
        logger.warning("  %s (%s): %s", os.path.basename(t), phase, err)
    if not results and old is None:
        raise RuntimeError("Keine Mine erfolgreich verarbeitet.")

    feats, polys = combine_results(results, tiles) if results else (None, None)
    if old is not None:
        parts = {"old": old}
        if feats is not None:
            parts["new"] = (feats, polys)
        feats = pd.concat([p[0] for p in parts.values()], ignore_index=True)
        crs = old[1].crs
        polys = gpd.GeoDataFrame(pd.concat([p[1].to_crs(crs) for p in parts.values()], ignore_index=True), crs=crs)
    fpath, _ = write_cache(feats, polys, cfg["output_dir"])
    logger.info("Zwischenspeicher geschrieben (%d Segmente): %s", len(feats), fpath)

    if args.train:
        from otr_obia_pipeline import main as pipeline_main
        pipeline_main({**cfg, "use_cache": True})
    else:
        logger.info("Weiter mit: python src/otr_obia_pipeline.py <gleiche Flags> --use-cache")
    return 1 if failed and not results else 0


if __name__ == "__main__":
    sys.exit(main())
