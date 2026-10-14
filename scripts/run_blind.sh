#!/bin/bash
# Blind test on the VPS (see scripts/blind_test.py): for each fold hide a group of confirmed dumps (by tile), retrain
# without them, score everything, build the candidate lists and count how many hidden dump places they contain.
#   bash scripts/run_blind.sh [FOLDS=3] [SUFFIX=all8]
cd /home/container || exit 1
export LD_LIBRARY_PATH=/home/container/libexpat_extracted/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PY=venv/bin/python3; BASE=out_px12_my_bright; BANDS=s2my_bright_early,s2my_bright_late
FOLDS=${1:-3}; SUF=${2:-all8}; TILE_CACHE=data/tile_cache
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
for K in $(seq 0 $((FOLDS-1))); do
  B=data/blind/f$K; RUN=out_blind_f$K; rm -rf $B $RUN; mkdir -p $B $RUN
  stamp "=== fold $K: prep"
  $PY scripts/blind_test.py prep --fold $K --folds $FOLDS --suffix $SUF --out $B || { stamp "prep FAILED"; exit 1; }
  for f in feature_cache.pkl polygons_cache.gpkg; do ln -s "$(realpath $BASE/$f)" $RUN/$f; done
  stamp "=== fold $K: retrain + rescore"
  $PY src/otr_obia_pipeline.py --imagery-dir $B/imagery --labels-path $B/dump_labels.gpkg --mine-boundary-path data/mine_boundaries.gpkg \
    --s1-dir data/all_s1 --dem-dir data/all_dem --s2my-dir data/s2_multiyear_all --s2my-bands $BANDS --target-segment-px 12 \
    --max-segments-per-mine 1000000 --n-jobs 12 --use-cache --tile-cache-dir $TILE_CACHE --model-cache-dir data/model_cache --labeled-imagery-dir $B/labeled --extra-labels-path $B/labels_extra.gpkg \
    --reviewed-imagery-dir $B/reviewed --extra-imagery-dir $B/extra \
    --add-labels-path $B/labels_reviewed_dump.gpkg --ignore-points-path $B/ignore_points.geojson,$B/ignore_reviewed.geojson \
    --hard-negatives-path $B/negatives_clean.gpkg --hard-negative-factor 20 \
    --osm-buildings-dir data/sites_quarries/osm_buildings --osm-roads-dir data/sites_quarries/osm_roads --osm-poi-dir data/sites_quarries/osm_poi --output-dir $RUN > log_blind_f$K.txt 2>&1 \
    || { stamp "pipeline FAILED"; tail -15 log_blind_f$K.txt; exit 1; }
  stamp "=== fold $K: candidate lists"
  COMMON="--segments $RUN/segments_classified.gpkg --preselect --area-pct 2 --labels $B/labels_for_eval.gpkg --exclude-known \
    --rank-by mean_proba --min-area-m2 2500 --max-site-m2 40000 --max-urban-frac 0.3 --exclude-areas $B/negatives_clean.gpkg,$B/ignore_reviewed.geojson \
    --operations data/mine_boundaries.gpkg,data/northChileQuarrys.geojson,data/chile_rest/ops_all.gpkg"
  $PY src/candidate_sites.py $COMMON --top 500 --out $RUN/cand_top500.gpkg > log_blind_f${K}_c1.txt 2>&1
  $PY src/candidate_sites.py $COMMON --top 2500 --out $RUN/cand_top2500.gpkg > log_blind_f${K}_c2.txt 2>&1
  $PY src/candidate_sites.py $COMMON --out $RUN/cand_all.gpkg > log_blind_f${K}_c3.txt 2>&1
  $PY src/eval_segment_size.py --labels $B/truth.gpkg --run blind=$RUN > $B/eval_segment.txt 2>&1
  $PY scripts/blind_test.py score --fold $K --out $B --lists $RUN/cand_top500.gpkg $RUN/cand_top2500.gpkg $RUN/cand_all.gpkg
done
$PY scripts/blind_test.py merge --glob "data/blind/f*/score.json" --out blind_results.json
stamp "=== BLIND TEST DONE ==="
