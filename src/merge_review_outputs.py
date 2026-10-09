"""
Neue Prüfergebnisse zu den bisherigen sammeln
=============================================

reviewed_to_labels.py schreibt pro geprüfter Datei vier Ausgaben. Dieses Skript hängt die Ausgaben
mehrerer Prüfrunden an die bisherigen Sammeldateien an (doppelte Geometrien werden entfernt) und
legt dazu die Liste der Kacheln an, in denen etwas geprüft wurde (für --reviewed-imagery-dir).

Beispiel:
    python src/merge_review_outputs.py --base-dir data/reviewed/out --suffix all --out-suffix all3 \\
        --new rev20/out --new rev5/out --eval-labels data/reviewed/out/labels_all2_for_eval.gpkg \\
        --tiles-out data/reviewed/out/reviewed_tiles_all3.txt

Erzeugt in --base-dir: labels_reviewed_dump_<out>.gpkg, negatives_<out>_clean.gpkg,
ignore_reviewed_<out>.geojson und (mit --eval-labels) labels_<out>_for_eval.gpkg.
"""
import argparse
import os
import sys

import geopandas as gpd
import pandas as pd


def _read(path):
    return gpd.read_file(path) if os.path.exists(path) else None


def concat_dedup(frames, crs=None):
    # ältere Sammeldateien (z.B. ignore_reviewed_all.geojson) haben nur Geometrien, keine mine_id
    frames = [f.assign(mine_id=f["mine_id"] if "mine_id" in f.columns else "")[["mine_id", "geometry"]].to_crs(crs or f.crs)
              for f in frames if f is not None and len(f)]
    if not frames:
        return gpd.GeoDataFrame({"mine_id": []}, geometry=[], crs=crs)
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    out["mine_id"] = out["mine_id"].astype(str)
    key = out["mine_id"] + "|" + out.geometry.apply(lambda g: g.normalize().wkb_hex)
    return out[~key.duplicated()].reset_index(drop=True)


def merge(base_dir, suffix, out_suffix, new_dirs, eval_labels=None, tiles_out=None):
    kinds = {
        "dump": (f"labels_reviewed_dump_{suffix}.gpkg", "labels_reviewed_dump.gpkg", f"labels_reviewed_dump_{out_suffix}.gpkg"),
        "neg": (f"negatives_{suffix}_clean.gpkg", "negatives_reviewed_clean.gpkg", f"negatives_{out_suffix}_clean.gpkg"),
        "ign": (f"ignore_reviewed_{suffix}.geojson", "ignore_reviewed.geojson", f"ignore_reviewed_{out_suffix}.geojson"),
    }
    merged, report = {}, {}
    for k, (base_name, new_name, out_name) in kinds.items():
        parts = [_read(os.path.join(base_dir, base_name))] + [_read(os.path.join(d, new_name)) for d in new_dirs]
        n_before = len(parts[0]) if parts[0] is not None else 0
        merged[k] = concat_dedup(parts)
        driver = "GeoJSON" if out_name.endswith(".geojson") else "GPKG"
        path = os.path.join(base_dir, out_name)
        if os.path.exists(path):
            os.remove(path)
        merged[k].to_file(path, driver=driver)
        report[k] = (n_before, len(merged[k]))
    if eval_labels:
        ev = concat_dedup([gpd.read_file(eval_labels), merged["dump"]])
        path = os.path.join(base_dir, f"labels_{out_suffix}_for_eval.gpkg")
        if os.path.exists(path):
            os.remove(path)
        ev.to_file(path, driver="GPKG")
        report["eval"] = (len(gpd.read_file(eval_labels)), len(ev))
    tiles = sorted(set(pd.concat([merged[k]["mine_id"] for k in merged]).astype(str)) - {""})
    if tiles_out:
        with open(tiles_out, "w") as f:
            f.write("\n".join(tiles) + "\n")
    return report, tiles


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True)
    p.add_argument("--suffix", default="all", help="Endung der bisherigen Sammeldateien.")
    p.add_argument("--out-suffix", required=True, help="Endung der neuen Sammeldateien.")
    p.add_argument("--new", action="append", default=[], help="Ausgabeordner von reviewed_to_labels.py (mehrfach).")
    p.add_argument("--eval-labels", default=None, help="Bisherige Bewertungs-Labels, um die neuen Halden ergänzt.")
    p.add_argument("--tiles-out", default=None, help="Textdatei mit den Kacheln, in denen etwas geprüft wurde.")
    a = p.parse_args(argv)
    report, tiles = merge(a.base_dir, a.suffix, a.out_suffix, a.new, a.eval_labels, a.tiles_out)
    for k, (before, after) in report.items():
        print(f"{k}: {before} -> {after}")
    print(f"{len(tiles)} Kacheln mit geprüften Stellen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
