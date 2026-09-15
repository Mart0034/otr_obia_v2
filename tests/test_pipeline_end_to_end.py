"""Runs the entire pipeline (segmentation -> labeling -> training ->
GeoPackage export) start to finish on made-up 'mines', without needing any
real satellite imagery. This is the safety net that catches the pipeline
breaking as a whole, even if every individual piece still passes its own
test."""
import geopandas as gpd
from shapely.geometry import box

from otr_obia_pipeline import CONFIG, main


def test_full_pipeline_runs_on_synthetic_mines(tmp_path, write_synthetic_raster):
    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()
    output_dir = tmp_path / "output"

    # Two mines in DIFFERENT UTM zones -> exercises the CRS-mixing bug fix.
    write_synthetic_raster(
        imagery_dir / "1.tif", seed=1, crs="EPSG:32632", nodata_corner=True
    )
    write_synthetic_raster(imagery_dir / "2.tif", seed=2, crs="EPSG:32633")

    # mine_id stored as an int on purpose -> exercises the text/number fix.
    dump_polygon = box(500000 + 50, 5600000 - 150, 500000 + 150, 5600000 - 50)
    labels_gdf = gpd.GeoDataFrame(
        {"mine_id": [1]}, geometry=[dump_polygon], crs="EPSG:32632"
    )
    labels_path = tmp_path / "labels.gpkg"
    labels_gdf.to_file(labels_path, driver="GPKG")

    cfg = dict(CONFIG)
    cfg.update(
        {
            "imagery_dir": str(imagery_dir),
            "labels_path": str(labels_path),
            "mine_id_field": "mine_id",
            "n_segments_per_mine": 30,
            "min_overlap_ratio": 0.3,
            "output_dir": str(output_dir),
        }
    )

    main(cfg)

    out_path = output_dir / "segments_classified.gpkg"
    assert out_path.exists()

    result = gpd.read_file(out_path)
    assert len(result) > 0
    assert "dump_proba" in result.columns
    assert "dump_pred" in result.columns

    # mine 1 has a digitized dump -> should end up with positive segments
    assert int(result[result.mine_id == "1"]["label"].sum()) > 0
    # mine 2 has no labels at all -> must never be marked positive
    assert int(result[result.mine_id == "2"]["label"].sum()) == 0
    # both mines ended up in one shared coordinate system despite starting
    # in different UTM zones -> guards against the CRS-mixing bug
    assert result.crs.to_string() == "EPSG:32632"
