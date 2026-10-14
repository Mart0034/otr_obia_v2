import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import transfer_eval as te  # noqa: E402


def test_evaluate_reports_score_and_percentile_near_a_point():
    x0, y0 = 500000.0, 4000000.0
    seg = gpd.GeoDataFrame({"mine_id": ["t1"] * 3, "dump_proba": [0.9, 0.1, 0.2]},
                           geometry=[box(x0, y0, x0 + 20, y0 + 20), box(x0 + 500, y0, x0 + 520, y0 + 20),
                                     box(x0 + 1000, y0, x0 + 1020, y0 + 20)], crs=32612)
    lon, lat = gpd.GeoSeries([box(x0, y0, x0 + 20, y0 + 20).centroid], crs=32612).to_crs(4326).iloc[0].coords[0]
    out = te.evaluate(pd.DataFrame({"name": ["a", "b"], "lat": [lat, 10.0], "lon": [lon, 10.0], "result": ["dump", "none"]}), seg)
    assert out.loc[0, "max_proba"] == 0.9 and out.loc[0, "pct_in_tile"] > 60
    assert out.loc[1, "segments"] == 0                                               # Ort ohne Segmente: kein Fehler
