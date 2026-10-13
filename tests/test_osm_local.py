import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import osm_local  # noqa: E402


def test_parse_other_tags():
    d = osm_local.parse_other_tags('"amenity"=>"cafe","name"=>"A \\"B\\"","shop"=>"kiosk"')
    assert d["amenity"] == "cafe" and d["shop"] == "kiosk"
    assert osm_local.parse_other_tags(None) == {}


def test_poi_kind_from_columns_and_other_tags():
    df = pd.DataFrame({
        "amenity": [None, "cafe", None, None, None],
        "landuse": [None, None, "retail", "forest", None],
        "other_tags": ['"tourism"=>"hotel"', None, None, None, '"name"=>"x"'],
    })
    kind = osm_local.poi_kind(df)
    assert kind.tolist() == ["amenity/shop", "amenity/shop", "landuse", None, None]


def test_clip_tile_writes_same_files_as_overpass_scripts(tmp_path):
    # kleine Kachel in UTM 19S bei 69.5 W / 23.5 S
    tif = tmp_path / "t_001.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32", crs="EPSG:32719",
                       transform=from_origin(500000, 7400000, 10, 10)) as dst:
        dst.write(np.zeros((10, 10), dtype="float32"), 1)
    lon, lat = gpd.GeoSeries([Point(500050, 7399950)], crs=32719).to_crs(4326).iloc[0].coords[0]
    bdir = tmp_path / "build"
    bdir.mkdir()
    gpd.GeoDataFrame(geometry=[LineString([(lon - 0.0002, lat), (lon + 0.0002, lat)]), LineString([(0, 0), (1, 1)])],
                     crs=4326).to_file(bdir / "roads.gpkg", driver="GPKG")
    gpd.GeoDataFrame(geometry=[box(lon - 0.0001, lat - 0.0001, lon + 0.0001, lat + 0.0001)], crs=4326).to_file(
        bdir / "buildings.gpkg", driver="GPKG")
    gpd.GeoDataFrame({"kind": ["amenity/shop"]}, geometry=[Point(10, 10)], crs=4326).to_file(bdir / "poi.gpkg", driver="GPKG")
    paths = {k: str(bdir / f"{k}.gpkg") for k in osm_local.LAYERS}
    counts = osm_local.clip_tile(paths, str(tif), str(tmp_path / "out"))
    assert counts == {"roads": 1, "buildings": 1, "poi": 0}
    roads = gpd.read_file(tmp_path / "out" / "osm_roads" / "t_001.gpkg")
    assert len(roads) == 1 and roads.crs.to_epsg() == 32719
    assert list(gpd.read_file(tmp_path / "out" / "osm_poi" / "t_001.gpkg").columns) == ["kind", "geometry"]
    # vorhandene Dateien bleiben stehen
    assert osm_local.clip_tile(paths, str(tif), str(tmp_path / "out")) == {"roads": None, "buildings": None, "poi": None}
