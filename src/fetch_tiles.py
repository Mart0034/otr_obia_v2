"""
Kacheln gleichzeitig aus allen Quellen laden (statt Quelle für Quelle)
=====================================================================

Bisher lief das Laden in Stufen: erst alle Sentinel-2-Bilder, dann alle Radarbilder, dann Höhenmodell, dann die
Mehrjahres-Helligkeit - jede Stufe wartete auf die vorige, obwohl Radar, Höhenmodell und Mehrjahres-Helligkeit nur
den Umriss der Kachel brauchen (nicht den Inhalt des Sentinel-2-Bildes). Dieses Modul lädt pro Kachel:

    Sentinel-2-Basisbild  ->  gleichzeitig: Sentinel-1, Höhenmodell, Mehrjahres-Helligkeit, (OSM aus lokaler Datei)

und arbeitet mehrere Kacheln parallel ab. So ist die Leitung ständig ausgelastet; gemessen skaliert das Laden von
6 auf 12 gleichzeitige Kacheln etwa 1,8-fach, darüber kaum noch.

Vorhandene Dateien werden übersprungen; jede Datei wird atomar geschrieben (.part -> endgültiger Name).
Eine Kachel gilt als vollständig, wenn Basisbild, Radar, Höhenmodell und Mehrjahres-Helligkeit existieren.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

STAGES = ("sentinel1", "dem", "s2my")


def _default_fetchers(catalog=None, crs="EPSG:32719"):
    """Die echten Lade-Funktionen mit den bisherigen Standardwerten der einzelnen Skripte."""
    import fetch_dem
    import fetch_s2_multiyear
    import fetch_sentinel1
    from fetch_s2_base_imagery import fetch_for_site

    catalog = catalog or fetch_sentinel1.open_catalog()
    years = list(range(2018, 2026))

    def base(tile, out_path):
        return fetch_for_site(tile["center_x"], tile["center_y"], tile["radius_m"], out_path, catalog,
                              crs=crs)

    return {
        "base": base,
        "sentinel1": lambda s2, out: fetch_sentinel1.fetch_for_mine(s2, out, catalog, "2023-01-01", "2023-12-31", 12, 3),
        "dem": lambda s2, out: fetch_dem.fetch_for_mine(s2, out, catalog),
        "s2my": lambda s2, out: fetch_s2_multiyear.fetch_for_mine(s2, out, catalog, years, 3, 15, 4, 3),
    }


def _atomic(out, fn):
    part = out + ".part"
    try:
        msg = fn(part)
        os.replace(part, out)
        return msg
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise


def fetch_tiles(tiles, out_dir, n_workers=8, inner_workers=3, fetchers=None, osm_clip=None, imagery_name="imagery",
                crs="EPSG:32719"):
    """tiles: Liste von dicts (site_id, center_x, center_y, radius_m). Lädt alles nach out_dir/{imagery,sentinel1,dem,s2my}.
    osm_clip(tif_path): optional, schneidet OSM aus einer lokalen Datei zu (siehe osm_local.py).
    fetchers: für Tests austauschbar. Gibt {"done": [...], "failed": {site_id: Grund}, "seconds": {Stufe: Summe}} zurück."""
    fetchers = fetchers or _default_fetchers(crs=crs)
    dirs = {"base": os.path.join(out_dir, imagery_name)}
    for stage in STAGES:
        dirs[stage] = os.path.join(out_dir, stage)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    seconds = {k: 0.0 for k in ("base",) + STAGES + ("osm",)}
    failed = {}

    def timed(stage, fn):
        t0 = time.time()
        try:
            return fn()
        finally:
            seconds[stage] += time.time() - t0

    def one(tile):
        sid = tile["site_id"]
        tif = os.path.join(dirs["base"], f"{sid}.tif")
        notes = []
        if not os.path.exists(tif):
            notes.append("Basis: " + timed("base", lambda: _atomic(tif, lambda part: fetchers["base"](tile, part))))
        todo = []
        for stage in STAGES:
            out = os.path.join(dirs[stage], f"{sid}.tif")
            if not os.path.exists(out):
                todo.append((stage, out))
        errors = []
        if todo:
            def run(job):
                stage, out = job
                return stage, timed(stage, lambda: _atomic(out, lambda part: fetchers[stage](tif, part)))
            with ThreadPoolExecutor(max_workers=max(1, inner_workers)) as pool:
                futs = {pool.submit(run, j): j[0] for j in todo}
                for fut in as_completed(futs):
                    try:
                        stage, msg = fut.result()
                        notes.append(f"{stage}: {msg}")
                    except Exception as e:
                        errors.append(f"{futs[fut]}: {e}")
        if osm_clip is not None:
            try:
                timed("osm", lambda: osm_clip(tif))
            except Exception as e:
                errors.append(f"osm: {e}")
        if errors:
            raise RuntimeError("; ".join(errors))
        return ", ".join(notes) or "vorhanden"

    t_start, done = time.time(), []
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        futures = {pool.submit(one, t): t["site_id"] for t in tiles}
        for i, fut in enumerate(as_completed(futures), 1):
            sid = futures[fut]
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(futures) - i)
            try:
                logger.info("  [%d/%d] %s: %s (noch ca. %dm%02ds)", i, len(futures), sid, fut.result(), eta // 60, eta % 60)
                done.append(sid)
            except Exception as e:
                logger.error("  [%d/%d] %s: fehlgeschlagen (%s)", i, len(futures), sid, e)
                failed[sid] = str(e)
    wall = time.time() - t_start
    logger.info("Fertig in %dm%02ds: %d von %d Kacheln vollständig (%d Kacheln und %d Teilaufgaben gleichzeitig). "
                "Summe der Einzelzeiten: %s", wall // 60, wall % 60, len(done), len(futures), n_workers, inner_workers,
                ", ".join(f"{k} {v:.0f} s" for k, v in seconds.items() if v))
    if failed:
        logger.warning("%d Kacheln unvollständig: %s", len(failed), sorted(failed))
    return {"done": sorted(done), "failed": failed, "seconds": seconds}
