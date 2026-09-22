"""Regression test: predict_and_export() must not crash when the final
model was trained on a single class (e.g. zero positive segments found
across the whole dataset) - this happened for real on the full 138-mine
dataset before segmentation was made size-aware."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier

from otr_obia_pipeline import predict_and_export


def test_predict_and_export_handles_single_class_model(tmp_path):
    feature_df = pd.DataFrame({
        "mine_id": ["1", "1", "2"],
        "segment_id": [1, 2, 1],
        "label": [0, 0, 0],
        "blue_mean": [10.0, 20.0, 30.0],
    })
    polygons_gdf = gpd.GeoDataFrame({
        "mine_id": ["1", "1", "2"],
        "segment_id": [1, 2, 1],
        "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)],
    }, crs="EPSG:32632")

    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(feature_df[["blue_mean"]].values, feature_df["label"].values)
    assert len(clf.classes_) == 1  # sanity check: only class 0 present

    cfg = {"output_dir": str(tmp_path)}
    out_path = predict_and_export(
        feature_df, polygons_gdf, clf, ["blue_mean"], cfg
    )

    result = gpd.read_file(out_path)
    assert (result["dump_proba"] == 0.0).all()
