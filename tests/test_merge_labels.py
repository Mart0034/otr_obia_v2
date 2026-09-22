"""Tests for merge_labels.py, using the same messy real-world schema we
actually found in Benedikt's data: some files empty, some with a
'dumpsite' column, some with a typo'd column name, some with none at all.
merge_labels() must not care about any of that, only the geometry."""
import geopandas as gpd
import pytest
from shapely.geometry import box

from merge_labels import merge_labels


def _write_shp(path, geoms=None, extra_cols=None):
    if geoms is None:
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        data = {"id": list(range(len(geoms)))}
        if extra_cols:
            data.update(extra_cols)
        gdf = gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:4326")
    gdf.to_file(path, driver="ESRI Shapefile")


def test_merges_mixed_schema_and_skips_empty_files(tmp_path):
    labels_dir = tmp_path / "Labels_qgis"
    labels_dir.mkdir()

    _write_shp(labels_dir / "mine_001.shp", geoms=None)  # confirmed no dump
    _write_shp(
        labels_dir / "mine_007.shp",
        geoms=[box(0, 0, 1, 1)],
        extra_cols={"dumpsite": ["yes"]},
    )
    _write_shp(
        labels_dir / "mine_012.shp",
        geoms=[box(2, 2, 3, 3), box(4, 4, 5, 5)],  # no attribute column at all
    )
    _write_shp(
        labels_dir / "mine_019.shp",
        geoms=[box(6, 6, 7, 7)],
        extra_cols={"dumspite": ["yes"]},  # typo'd column, seen in the real data
    )

    merged = merge_labels(str(labels_dir))

    assert set(merged["mine_id"]) == {"mine_007", "mine_012", "mine_019"}
    assert len(merged) == 4  # 1 + 2 + 1 polygons; mine_001 contributes none
    assert (merged["mine_id"] == "mine_012").sum() == 2


def test_raises_if_no_shapefiles_found(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        merge_labels(str(empty_dir))


def test_raises_if_every_file_is_empty(tmp_path):
    labels_dir = tmp_path / "Labels_qgis"
    labels_dir.mkdir()
    _write_shp(labels_dir / "mine_001.shp", geoms=None)
    _write_shp(labels_dir / "mine_002.shp", geoms=None)

    with pytest.raises(ValueError):
        merge_labels(str(labels_dir))


def test_mine_id_field_name_is_configurable(tmp_path):
    labels_dir = tmp_path / "Labels_qgis"
    labels_dir.mkdir()
    _write_shp(labels_dir / "mine_025.shp", geoms=[box(0, 0, 1, 1)])

    merged = merge_labels(str(labels_dir), mine_id_field="site_id")

    assert "site_id" in merged.columns
    assert merged.loc[0, "site_id"] == "mine_025"
