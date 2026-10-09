"""Tests for the shared fetch helpers (atomic writes, failure markers, parallel reads) and the
fetch+process streaming driver."""
import os
import threading
import time

import numpy as np
import pandas as pd
import pytest
import rasterio
from concurrent.futures import ThreadPoolExecutor
from rasterio.transform import from_origin

import fetch_s2_multiyear
import fetch_sentinel1
from fetch_sentinel1 import _fail_marker, fetch_one, parallel_map, run_for_all_mines
from stream_pipeline import combine_results, run_stream, write_cache


def _tile(path):
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
                       crs="EPSG:32719", transform=from_origin(0, 40, 10, 10)) as dst:
        dst.write(np.ones((1, 4, 4), dtype="float32"))


def test_parallel_map_keeps_order_and_runs_concurrently():
    seen = set()

    def slow(x):
        seen.add(threading.get_ident())
        time.sleep(0.05)
        return x * 2

    assert parallel_map(slow, range(6), 3) == [0, 2, 4, 6, 8, 10]
    assert len(seen) > 1
    assert parallel_map(lambda x: x + 1, [1, 2], 1) == [2, 3]


def test_interrupted_write_leaves_no_file_and_is_retried(tmp_path):
    out = str(tmp_path / "a.tif")

    def boom(src, dst):
        open(dst, "w").write("half")
        raise OSError("network dropped")

    with pytest.raises(OSError):
        fetch_one("x", out, boom)
    assert not os.path.exists(out) and not os.path.exists(out + ".part")
    assert not os.path.exists(_fail_marker(out))          # transient: try again next time
    msg, dur = fetch_one("x", out, lambda s, d: (open(d, "w").write("ok"), "fine")[1])
    assert msg == "fine" and os.path.exists(out)


def test_missing_data_is_remembered_and_skipped_until_retry(tmp_path):
    img, out = tmp_path / "img", tmp_path / "out"
    img.mkdir()
    _tile(str(img / "mine_001.tif"))
    calls = []

    def nodata(src, dst):
        calls.append(src)
        raise RuntimeError("keine Aufnahmen")

    run_for_all_mines(str(img), str(out), 1, False, nodata)
    assert os.path.exists(_fail_marker(str(out / "mine_001.tif")))
    run_for_all_mines(str(img), str(out), 1, False, nodata)
    assert len(calls) == 1                                    # not retried
    run_for_all_mines(str(img), str(out), 1, False, nodata, retry_failed=True)
    assert len(calls) == 2


def test_existing_outputs_are_skipped_and_stale_parts_removed(tmp_path):
    img, out = tmp_path / "img", tmp_path / "out"
    img.mkdir(); out.mkdir()
    _tile(str(img / "mine_001.tif")); _tile(str(img / "mine_002.tif"))
    (out / "mine_001.tif").write_text("done")
    (out / "mine_002.tif.part").write_text("stale")
    seen = []
    run_for_all_mines(str(img), str(out), 2, False,
                      lambda s, d: (seen.append(os.path.basename(s)), open(d, "w").write("n"), "ok")[2])
    assert seen == ["mine_002.tif"]
    assert (out / "mine_001.tif").read_text() == "done"
    assert not (out / "mine_002.tif.part").exists()


def test_multiyear_fetch_assembles_years_from_parallel_reads(tmp_path, monkeypatch):
    s2 = str(tmp_path / "mine_001.tif"); _tile(s2)
    years = [2018, 2019, 2024, 2025]

    class Item:
        def __init__(self, year):
            self.year = year
            self.datetime = None
            self.id = f"{year}"
            self.properties = {"eo:cloud_cover": 1, "datetime": f"{year}-06-01T00:00:00Z",
                               "s2:processing_baseline": "04.00"}
            self.assets = {b: type("A", (), {"href": f"{year}|{b}"})() for b in
                           ("B02", "B03", "B04", "B08", "SCL")}

    class Cat:
        def search(self, collections, bbox, datetime):
            year = int(datetime[:4])
            return type("S", (), {"items": lambda self: [Item(year)]})()

    def fake_read(href, crs, transform, w, h, resampling=None, **kw):
        year, band = href.split("|")
        if band == "SCL":
            return np.full((h, w), 4, dtype=np.float32)
        # raw DN: brighter before 2024, darker after (brightness ~ DN/10000 - offset handled by to_reflectance)
        return np.full((h, w), 4000 if int(year) < 2024 else 1500, dtype=np.float32)

    monkeypatch.setattr(fetch_s2_multiyear, "_read_on_grid", fake_read)
    monkeypatch.setattr(fetch_s2_multiyear, "_dedupe_by_date", lambda items: items)
    out = str(tmp_path / "my.tif")
    msg = fetch_s2_multiyear.fetch_for_mine(s2, out, Cat(), years, 3, 15, 3, read_threads=4)
    assert "4 von 4" in msg
    with rasterio.open(out) as r:
        early, late, delta = r.read(1)[0, 0], r.read(2)[0, 0], r.read(3)[0, 0]
    assert late < early and delta == pytest.approx(late - early, abs=1e-4)


def test_stream_processes_tiles_while_others_still_download(tmp_path):
    tiles = []
    for i in range(4):
        p = tmp_path / f"mine_{i}.tif"; _tile(str(p)); tiles.append(str(p))
    events = []

    def fetch(t):
        time.sleep(0.15 * (tiles.index(t) + 1))
        events.append(("fetched", os.path.basename(t)))
        if t.endswith("mine_2.tif"):
            raise RuntimeError("no data")

    def process(t):
        events.append(("processed", os.path.basename(t)))
        return os.path.basename(t)

    results, failed = run_stream(tiles, fetch, process, 4, lambda n: ThreadPoolExecutor(n or 2))
    assert sorted(results.values()) == ["mine_0.tif", "mine_1.tif", "mine_3.tif"]
    assert [os.path.basename(t) for t, ph, _ in failed] == ["mine_2.tif"] and failed[0][1] == "download"
    # the first tile was processed before the last one finished downloading
    assert events.index(("processed", "mine_0.tif")) < events.index(("fetched", "mine_3.tif"))


def test_stream_retries_crashed_pool_tiles_one_by_one(tmp_path):
    from concurrent.futures.process import BrokenProcessPool
    t = tmp_path / "mine_0.tif"; _tile(str(t))
    state = {"n": 0}

    class FlakyExec(ThreadPoolExecutor):
        def submit(self, fn, *a):
            fut = super().submit(fn, *a)
            state["n"] += 1
            if state["n"] == 1:
                f2 = super().submit(lambda: (_ for _ in ()).throw(BrokenProcessPool("oom")))
                return f2
            return fut

    results, failed = run_stream([str(t)], lambda x: None, lambda x: "done", 1, lambda n: FlakyExec(n or 1))
    assert results == {str(t): "done"} and not failed


def test_combine_and_atomic_cache_write(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    def part(mid):
        f = pd.DataFrame({"mine_id": [mid], "segment_id": [0], "label": [0], "x": [1.0]})
        g = gpd.GeoDataFrame({"mine_id": [mid], "segment_id": [0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32719")
        return f, g
    res = {"a": part("a"), "b": part("b")}
    feats, polys = combine_results(res, ["b", "a"])
    assert list(feats["mine_id"]) == ["b", "a"] and len(polys) == 2
    fp, pp = write_cache(feats, polys, str(tmp_path / "o"))
    assert os.path.exists(fp) and os.path.exists(pp)
    assert not [f for f in os.listdir(tmp_path / "o") if "tmp" in f]


def test_streamed_tiles_give_the_same_dataset_as_the_normal_build(tmp_path, write_synthetic_raster):
    import geopandas as gpd
    from shapely.geometry import box
    from otr_obia_pipeline import CONFIG, build_dataset
    from stream_pipeline import process_tile

    img = tmp_path / "imagery"; img.mkdir()
    write_synthetic_raster(img / "1.tif", seed=1)
    write_synthetic_raster(img / "2.tif", seed=2)
    lab = tmp_path / "labels.gpkg"
    gpd.GeoDataFrame({"mine_id": [1]}, geometry=[box(500050, 5599850, 500150, 5599950)],
                     crs="EPSG:32632").to_file(lab, driver="GPKG")
    cfg = {**CONFIG, "imagery_dir": str(img), "labels_path": str(lab), "n_segments_per_mine": 20,
           "target_segment_px": None, "min_overlap_ratio": 0.3, "n_jobs": 1}
    ref_f, ref_p = build_dataset(cfg)
    res = {str(img / n): process_tile(str(img / n), cfg) for n in ("1.tif", "2.tif")}
    feats, polys = combine_results(res, [str(img / "1.tif"), str(img / "2.tif")])
    key = ["mine_id", "segment_id"]
    pd.testing.assert_frame_equal(
        feats.sort_values(key).reset_index(drop=True), ref_f.sort_values(key).reset_index(drop=True))
    assert len(polys) == len(ref_p)


def test_stream_main_builds_cache_and_resume_skips_known_mines(tmp_path, write_synthetic_raster):
    import geopandas as gpd
    from shapely.geometry import box
    from stream_pipeline import main

    img = tmp_path / "imagery"; img.mkdir()
    write_synthetic_raster(img / "1.tif", seed=1)
    write_synthetic_raster(img / "2.tif", seed=2)
    lab = tmp_path / "labels.gpkg"
    gpd.GeoDataFrame({"mine_id": [1]}, geometry=[box(500050, 5599850, 500150, 5599950)],
                     crs="EPSG:32632").to_file(lab, driver="GPKG")
    out = tmp_path / "out"
    argv = ["--imagery-dir", str(img), "--labels-path", str(lab), "--output-dir", str(out),
            "--n-segments-per-mine", "20", "--target-segment-px", "0", "--n-jobs", "2"]
    assert main(argv) == 0
    df = pd.read_pickle(out / "feature_cache.pkl")
    assert set(df["mine_id"]) == {"1", "2"}
    write_synthetic_raster(img / "3.tif", seed=3)
    assert main(argv + ["--resume"]) == 0
    df2 = pd.read_pickle(out / "feature_cache.pkl")
    assert set(df2["mine_id"]) == {"1", "2", "3"} and len(df2) > len(df)
