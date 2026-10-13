import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fetch_tiles  # noqa: E402


def _fetchers(log, fail=None):
    lock = threading.Lock()

    def make(stage):
        def fn(*args):
            out = args[-1]
            time.sleep(0.01)
            with lock:
                log.append((stage, os.path.basename(out)))
            if fail == stage:
                raise RuntimeError(f"{stage} kaputt")
            open(out, "w").write(stage)
            return f"{stage} ok"
        return fn
    return {k: make(k) for k in ("base", "sentinel1", "dem", "s2my")}


def test_all_sources_per_tile_base_first(tmp_path):
    log = []
    tiles = [{"site_id": f"t{i}", "center_x": 0, "center_y": 0, "radius_m": 500} for i in range(5)]
    res = fetch_tiles.fetch_tiles(tiles, str(tmp_path), n_workers=3, inner_workers=3, fetchers=_fetchers(log))
    assert res["done"] == [f"t{i}" for i in range(5)] and not res["failed"]
    for i in range(5):
        for d in ("imagery", "sentinel1", "dem", "s2my"):
            assert (tmp_path / d / f"t{i}.tif").exists()
        order = [s for s, f in log if f.startswith(f"t{i}.tif")]
        assert order[0] == "base" and set(order[1:]) == {"sentinel1", "dem", "s2my"}   # Basisbild zuerst
    assert not list(tmp_path.rglob("*.part"))


def test_existing_files_are_skipped_and_failures_reported(tmp_path):
    log = []
    tiles = [{"site_id": "a", "center_x": 0, "center_y": 0, "radius_m": 500}]
    fetch_tiles.fetch_tiles(tiles, str(tmp_path), fetchers=_fetchers(log))
    n = len(log)
    fetch_tiles.fetch_tiles(tiles, str(tmp_path), fetchers=_fetchers(log))
    assert len(log) == n                                   # nichts erneut geladen
    (tmp_path / "dem" / "a.tif").unlink()
    log2 = []
    res = fetch_tiles.fetch_tiles(tiles, str(tmp_path), fetchers=_fetchers(log2, fail="dem"))
    assert [s for s, _ in log2] == ["dem"] and "a" in res["failed"] and not res["done"]
    assert not (tmp_path / "dem" / "a.tif").exists() and not list(tmp_path.rglob("*.part"))
