"""Eine Mine allein segmentieren und dabei alle 2 Minuten ausgeben, WO im Code
das Programm gerade steckt (faulthandler) - zum Aufspüren von Speicher-/Laufzeitproblemen.

Nutzung:
    python src/profile_mine.py data/imagery/mine_019.tif 30
"""
import faulthandler
import logging
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
from otr_obia_pipeline import CONFIG, build_dataset  # noqa: E402


def main(tif, px):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    faulthandler.dump_traceback_later(120, repeat=True)
    with tempfile.TemporaryDirectory() as d:
        os.symlink(os.path.abspath(tif), os.path.join(d, os.path.basename(tif)))
        cfg = dict(CONFIG)
        cfg.update(imagery_dir=d, target_segment_px=px, max_segments_per_mine=1000000, n_jobs=1,
                   labels_path="data/dump_labels.gpkg", mine_boundary_path="data/mine_boundaries.gpkg",
                   s1_dir="data/sentinel1", dem_dir="data/dem")
        t0 = time.time()
        feats, _ = build_dataset(cfg)
        print(f"FERTIG: {len(feats)} Segmente in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
