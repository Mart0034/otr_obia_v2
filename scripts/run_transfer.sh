#!/bin/bash
# Transfer test: score the Chile model at places outside Chile that a person has checked (reference/external_checks.csv).
#   bash scripts/run_transfer.sh [NAME=transfer] [CRS=EPSG:32612] [WAIT_FOR_FILE=blind.log WAIT_FOR_TEXT="BLIND TEST DONE"]
# One 2 km tile per place, all in one metric CRS. Uses the cached Chile model; exports only the new tiles.
cd /home/container || exit 1
export LD_LIBRARY_PATH=/home/container/libexpat_extracted/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PY=venv/bin/python3; OUT=data/reviewed/out; Q=data/sites_quarries; BASE=out_px12_my_bright; BANDS=s2my_bright_early,s2my_bright_late
NAME=${1:-transfer}; CRS=${2:-EPSG:32612}; WF=${3:-}; WT=${4:-}; LABELS=all8
R=data/$NAME; RUN=out_$NAME; mkdir -p $R
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
if [ -n "$WF" ]; then stamp "waiting for '$WT' in $WF"; until grep -q "$WT" $WF 2>/dev/null; do sleep 60; done; fi
stamp "=== 1. tiles around the checked places"
$PY - <<P
import pandas as pd
from pyproj import Transformer
c = pd.read_csv("reference/external_checks.csv"); t = Transformer.from_crs("EPSG:4326", "$CRS", always_xy=True)
x, y = t.transform(c["lon"].values, c["lat"].values)
pd.DataFrame({"site_id": [f"tx_{i:02d}" for i in range(len(c))], "center_x": x.round(-1), "center_y": y.round(-1), "radius_m": 1000.0}).to_csv("$R/tiles.csv", index=False)
print(len(c), "tiles")
P
$PY - <<P
import sys; sys.path.insert(0, "src")
import pandas as pd
from fetch_tiles import fetch_tiles
res = fetch_tiles(pd.read_csv("$R/tiles.csv").to_dict("records"), "$R/tiles", 6, 3, crs="$CRS")
print({"done": len(res["done"]), "failed": res["failed"]})
P
stamp "=== 2. link tiles with all inputs"
mkdir -p data/all_s1 data/all_dem data/s2_multiyear_all $R/ok_imagery
for f in $R/tiles/imagery/*.tif; do id=$(basename $f .tif)
  if [ -e $R/tiles/sentinel1/$id.tif ] && [ -e $R/tiles/dem/$id.tif ] && [ -e $R/tiles/s2my/$id.tif ]; then
    ln -sf "$(realpath $f)" $R/ok_imagery/$id.tif; ln -sf "$(realpath $R/tiles/sentinel1/$id.tif)" data/all_s1/$id.tif
    ln -sf "$(realpath $R/tiles/dem/$id.tif)" data/all_dem/$id.tif; ln -sf "$(realpath $R/tiles/s2my/$id.tif)" data/s2_multiyear_all/$id.tif
  fi; done
stamp "tiles ready: $(ls $R/ok_imagery | wc -l)"
stamp "=== 3. score with the cached Chile model"
rm -rf $RUN; mkdir $RUN; for f in feature_cache.pkl polygons_cache.gpkg; do ln -s "$(realpath $BASE/$f)" $RUN/$f; done
$PY src/otr_obia_pipeline.py --imagery-dir data/imagery --labels-path data/dump_labels.gpkg --mine-boundary-path data/mine_boundaries.gpkg \
  --s1-dir data/all_s1 --dem-dir data/all_dem --s2my-dir data/s2_multiyear_all --s2my-bands $BANDS --target-segment-px 12 \
  --max-segments-per-mine 1000000 --n-jobs 12 --use-cache --tile-cache-dir data/tile_cache --model-cache-dir data/model_cache \
  --labeled-imagery-dir data/run15/labeled --extra-labels-path data/run15/labels_extra.gpkg \
  --reviewed-imagery-dir $Q/imagery_reviewed --extra-imagery-dir $R/ok_imagery --export-extra-only --skip-polygons-extra \
  --add-labels-path $OUT/labels_reviewed_dump_$LABELS.gpkg --ignore-points-path data/run15/ignore_points.geojson,$OUT/ignore_reviewed_$LABELS.geojson \
  --hard-negatives-path $OUT/negatives_${LABELS}_clean.gpkg --hard-negative-factor 20 --output-dir $RUN > log_$NAME.txt 2>&1 \
  || { stamp "pipeline FAILED"; tail -15 log_$NAME.txt; exit 1; }
stamp "=== 4. evaluate"
$PY scripts/transfer_eval.py --checks reference/external_checks.csv --segments $RUN/segments_classified.gpkg --out $RUN/transfer_eval.csv | tee $RUN/transfer_eval.txt
stamp "=== TRANSFER TEST DONE ==="
