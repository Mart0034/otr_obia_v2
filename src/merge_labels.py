"""Merge per-mine label shapefiles into one file the pipeline can read.

The real data (received via email from Benedikt) comes as one shapefile
per mine (e.g. Labels_qgis/mine_007.shp) instead of the single combined
GeoPackage CONFIG["labels_path"] expects. This script merges them into
one file, deriving each mine_id from the filename with the SAME logic
the main pipeline uses to read mine_id from imagery filenames
(extract_mine_id), so imagery and labels always match up.

Usage:
    python3 src/merge_labels.py --labels-dir data/Labels_qgis --output data/dump_labels.gpkg
"""
import argparse
import glob
import logging
import os

import geopandas as gpd
import pandas as pd

from otr_obia_pipeline import extract_mine_id

logger = logging.getLogger(__name__)


def merge_labels(labels_dir, mine_id_field="mine_id"):
    """Reads every *.shp in labels_dir and merges the non-empty ones into
    a single GeoDataFrame with a mine_id column. Mines whose shapefile has
    zero features (a confirmed "no dump") are simply not included, exactly
    like the pipeline's existing "no labels for this mine -> label 0"
    behaviour already handles."""
    shp_paths = sorted(glob.glob(os.path.join(labels_dir, "*.shp")))
    if not shp_paths:
        raise FileNotFoundError(f"Keine Shapefiles in {labels_dir} gefunden.")

    frames = []
    n_empty = 0

    for path in shp_paths:
        mine_id = extract_mine_id(path)
        gdf = gpd.read_file(path)

        if gdf.empty:
            n_empty += 1
            continue

        # Nur die Geometrie behalten: die Attribut-Schemas unterscheiden
        # sich zwischen den Dateien (dumpsite/dumspite/dumpsites/fehlend),
        # aber label_segments() nutzt ohnehin nur die Geometrie, nie eine
        # Attributspalte, also spielt das keine Rolle.
        gdf = gdf[["geometry"]].copy()
        gdf[mine_id_field] = mine_id
        frames.append(gdf)

    if not frames:
        raise ValueError(f"Keine einzige Label-Datei in {labels_dir} enthält Polygone.")

    # Alle Dateien sollten dasselbe CRS haben, aber zur Sicherheit auf das
    # CRS der ersten nicht-leeren Datei vereinheitlichen, falls doch nicht.
    target_crs = frames[0].crs
    for i, gdf in enumerate(frames):
        if gdf.crs != target_crs:
            logger.warning(
                "%s hat ein anderes CRS (%s) als die übrigen Dateien (%s), reprojiziere.",
                shp_paths[i], gdf.crs, target_crs,
            )
            frames[i] = gdf.to_crs(target_crs)

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=target_crs)

    logger.info(
        "%d Shapefiles gelesen: %d Minen mit Dump-Polygonen (%d Polygone gesamt), "
        "%d Minen leer (bestätigt ohne Dump).",
        len(shp_paths), merged[mine_id_field].nunique(), len(merged), n_empty,
    )
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Führt die per-Mine Label-Shapefiles zu einer GeoPackage-Datei zusammen."
    )
    parser.add_argument(
        "--labels-dir", required=True, metavar="DIR",
        help="Ordner mit den mine_XXX.shp Dateien (z.B. Labels_qgis).",
    )
    parser.add_argument(
        "--output", default="data/dump_labels.gpkg", metavar="PATH",
        help="Wohin die zusammengeführte GeoPackage-Datei geschrieben wird.",
    )
    parser.add_argument(
        "--mine-id-field", default="mine_id", metavar="SPALTE",
        help="Name der Minen-ID-Spalte in der Ausgabedatei (muss zu "
             "CONFIG['mine_id_field'] passen).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    merged = merge_labels(args.labels_dir, args.mine_id_field)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_file(args.output, driver="GPKG")
    logger.info(
        "Geschrieben: %s (%d Polygone, %d Minen mit Dump)",
        args.output, len(merged), merged[args.mine_id_field].nunique(),
    )


if __name__ == "__main__":
    main()
