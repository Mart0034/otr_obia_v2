"""Schneller Sanity-Check der Eingabedaten, BEVOR die komplette (langsame)
Pipeline läuft. Prüft: Bandanzahl je GeoTIFF, Anteil gültiger (Nicht-NoData)
Pixel, und ob die aus den Dateinamen erratenen Minen-IDs tatsächlich zu den
IDs in der Label-Datei passen.

Nutzung:
    python3 src/check_data.py --imagery-dir data/imagery --labels-path data/dump_labels.gpkg
"""
import glob
import logging
import os

import geopandas as gpd
import numpy as np

from cli import build_arg_parser, resolve_config
from otr_obia_pipeline import CONFIG, extract_mine_id, load_mine_raster

logger = logging.getLogger(__name__)


def check_data(cfg):
    """Führt alle Prüfungen aus und gibt True zurück, wenn alles passt."""
    ok = True

    if not os.path.exists(cfg["labels_path"]):
        logger.error("Datei nicht gefunden: %s", cfg["labels_path"])
        return False

    labels_gdf = gpd.read_file(cfg["labels_path"])
    if cfg["mine_id_field"] not in labels_gdf.columns:
        logger.error(
            "Spalte '%s' nicht in %s gefunden. Vorhandene Spalten: %s",
            cfg["mine_id_field"], cfg["labels_path"], list(labels_gdf.columns),
        )
        return False

    label_mine_ids = set(labels_gdf[cfg["mine_id_field"]].astype(str).str.strip())
    logger.info(
        "%d Label-Polygone gefunden, Minen-IDs in labels_path: %s",
        len(labels_gdf), sorted(label_mine_ids),
    )

    imagery_paths = sorted(glob.glob(os.path.join(cfg["imagery_dir"], "*.tif")))
    if not imagery_paths:
        logger.error("Keine GeoTIFFs in %s gefunden.", cfg["imagery_dir"])
        return False
    logger.info("%d GeoTIFFs in %s gefunden.", len(imagery_paths), cfg["imagery_dir"])

    imagery_mine_ids = set()
    n_matched = 0
    for path in imagery_paths:
        mine_id = extract_mine_id(path)
        imagery_mine_ids.add(mine_id)

        try:
            arr, _transform, crs, nodata = load_mine_raster(
                path, expected_n_bands=len(cfg["band_names"])
            )
        except ValueError as exc:
            logger.error("%s: %s", os.path.basename(path), exc)
            ok = False
            continue

        valid_mask = np.all(np.isfinite(arr), axis=-1)
        if nodata is not None:
            valid_mask &= ~np.all(arr == nodata, axis=-1)
        valid_pct = 100 * float(valid_mask.mean())

        matched = mine_id in label_mine_ids
        if matched:
            n_matched += 1
        # Kein Label-Eintrag ist hier KEIN Fehler: eine Mine ohne bestätigten
        # Dump hat bei diesem Workflow bewusst keinen Eintrag in labels_path
        # (siehe merge_labels.py), und die Pipeline behandelt das bereits
        # korrekt als label=0. Nur wenn WIRKLICH GAR KEINE Mine matcht, ist
        # das ein Hinweis auf ein echtes Namensschema-Problem (siehe unten).
        status = "DUMP-LABEL" if matched else "kein Dump-Label"
        logger.info(
            "  %s (mine_id='%s'): %d Bänder, crs=%s, gültige Pixel=%.1f%%  [%s]",
            os.path.basename(path), mine_id, arr.shape[-1], crs, valid_pct, status,
        )
        if valid_pct < 50:
            logger.warning(
                "  -> nur %.1f%% gültige Pixel, evtl. falsches NoData oder Bildfehler.",
                valid_pct,
            )

    if label_mine_ids and imagery_mine_ids and n_matched == 0:
        logger.error(
            "Keine einzige Minen-ID aus %s stimmt mit einer Minen-ID aus den "
            "Bilddateien überein. Das deutet auf ein grundsätzliches Problem "
            "mit dem Namensschema hin (z.B. Text- vs. Zahlen-IDs), nicht "
            "einfach auf lauter negative Minen. Bitte Namensschema prüfen.",
        )
        ok = False
    elif imagery_mine_ids:
        logger.info(
            "%d von %d Minen haben ein bestätigtes Dump-Label, der Rest wird "
            "als negativ behandelt.", n_matched, len(imagery_mine_ids),
        )

    orphan_labels = label_mine_ids - imagery_mine_ids
    if orphan_labels:
        logger.warning(
            "Diese Minen-IDs haben Labels, aber kein passendes Bild: %s",
            sorted(orphan_labels),
        )

    if ok:
        logger.info("Alles passt, die Pipeline kann gestartet werden.")
    else:
        logger.error("Es gibt Probleme (siehe oben). Vor dem Pipeline-Lauf beheben.")

    return ok


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_arg_parser(
        "Prüft die Eingabedaten (Bilder + Labels), bevor die komplette Pipeline läuft."
    )
    cli_args = parser.parse_args()
    cfg = resolve_config(CONFIG, cli_args)
    success = check_data(cfg)
    raise SystemExit(0 if success else 1)
