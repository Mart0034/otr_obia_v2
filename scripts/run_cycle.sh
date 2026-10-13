#!/bin/bash
# One review-and-retrain cycle in a single command (run on the VPS in /home/container).
#
#   bash scripts/run_cycle.sh TAG --reviewed rev_in/my_reviews.gpkg --prev all4 --new all5 [--compare out_px12_qv4]
#       [--extra-root DIR] [--ops-extra FILE]   # more tile folders (made by sites_from_vector.py) and their mine/quarry outlines
#   bash scripts/run_cycle.sh TAG --labels all4 [--compare out_px12_qv4]       # same labels, e.g. a speed test
#
# What it does: reviews -> labels (reviewed_to_labels, merged with the earlier labels), splits the quarry tiles
# into reviewed / rest, retrains and rescores (per-tile feature cache, parallel road density), builds the two
# candidate lists (top 500 and every 5th of the top 2500, both with the `context` column), compares with an
# earlier run on the 138 known mines, and prints how long each phase took.
#
# Output: out_px12_q<TAG>/ (segments_classified.gpkg, candidate_sites_<TAG>_top500.gpkg / _sample5.gpkg),
# results_<TAG>.txt, log_q<TAG>.txt, timing_<TAG>.txt, and (with --reviewed) the labels *_<new>.*
cd /home/container || exit 1
export LD_LIBRARY_PATH=/home/container/libexpat_extracted/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PY=venv/bin/python3
OUT=data/reviewed/out
BASE=out_px12_my_bright
BANDS=s2my_bright_early,s2my_bright_late
Q=data/sites_quarries
TILE_CACHE=data/tile_cache

TAG=$1; shift
REVIEWED=""; PREV=""; NEW=""; LABELS=""; COMPARE=""; EXTRA_ROOT=""; OPS_EXTRA=""; CACHE_FLAG="--tile-cache-dir $TILE_CACHE"
while [ $# -gt 0 ]; do
  case $1 in
    --reviewed) REVIEWED=$2; shift 2;;
    --prev) PREV=$2; shift 2;;
    --new) NEW=$2; shift 2;;
    --labels) LABELS=$2; shift 2;;
    --compare) COMPARE=$2; shift 2;;
    --extra-root) EXTRA_ROOT=$2; shift 2;;
    --ops-extra) OPS_EXTRA=$2; shift 2;;
    --no-cache) CACHE_FLAG=""; shift;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -z "$TAG" ] && { echo "usage: run_cycle.sh TAG (--reviewed FILE --prev SUFFIX --new SUFFIX | --labels SUFFIX) [--compare RUNDIR]"; exit 2; }
if [ -n "$REVIEWED" ]; then
  [ -z "$PREV" ] || [ -z "$NEW" ] && { echo "--reviewed needs --prev and --new"; exit 2; }
  LABELS=$NEW
fi
[ -z "$LABELS" ] && { echo "need --reviewed ... or --labels SUFFIX"; exit 2; }

RUN=out_px12_q$TAG
T0=$(date +%s); LAST=$T0; : > timing_$TAG.txt
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
phase() { local now=$(date +%s); printf "%-48s %4d s\n" "$1" $((now-LAST)) | tee -a timing_$TAG.txt; LAST=$now; }
set -o pipefail

stamp "=== 0. folders with Sentinel-1 / DEM for every new tile (only missing links are added)"
mkdir -p data/all_s1 data/all_dem
for d in data/run15/labeled $Q/imagery; do
  for f in $d/*.tif; do
    id=$(basename $f)
    [ -e data/all_s1/$id ] || for s in $Q/sentinel1 data/run15/s1 data/boss/sentinel1; do [ -e $s/$id ] && { ln -sf "$(realpath $s/$id)" data/all_s1/$id; break; }; done
    [ -e data/all_dem/$id ] || for s in $Q/dem data/run15/dem data/boss/dem; do [ -e $s/$id ] && { ln -sf "$(realpath $s/$id)" data/all_dem/$id; break; }; done
  done
done
# extra tiles (new regions): keep only tiles that have all inputs, link them into the shared folders
TILES_ALL=$Q/imagery
if [ -n "$EXTRA_ROOT" ]; then
  rm -rf data/tiles_all; mkdir -p data/tiles_all $EXTRA_ROOT/complete
  for f in $Q/imagery/*.tif; do ln -sf "$(realpath $f)" data/tiles_all/$(basename $f); done
  n=0
  for f in $EXTRA_ROOT/imagery/*.tif; do id=$(basename $f .tif)
    if [ -e $EXTRA_ROOT/sentinel1/$id.tif ] && [ -e $EXTRA_ROOT/dem/$id.tif ] && [ -e $EXTRA_ROOT/s2my/$id.tif ]; then
      ln -sf "$(realpath $f)" data/tiles_all/$id.tif
      ln -sf "$(realpath $EXTRA_ROOT/sentinel1/$id.tif)" data/all_s1/$id.tif
      ln -sf "$(realpath $EXTRA_ROOT/dem/$id.tif)" data/all_dem/$id.tif
      ln -sf "$(realpath $EXTRA_ROOT/s2my/$id.tif)" data/s2_multiyear_all/$id.tif
      for k in roads buildings poi; do [ -e $EXTRA_ROOT/osm_$k/$id.gpkg ] && ln -sf "$(realpath $EXTRA_ROOT/osm_$k/$id.gpkg)" $Q/osm_$k/$id.gpkg; done
      n=$((n+1))
    fi
  done
  TILES_ALL=data/tiles_all
  stamp "extra tiles with all inputs: $n (from $EXTRA_ROOT)"
fi
stamp "all_s1: $(ls data/all_s1 | wc -l) files, all_dem: $(ls data/all_dem | wc -l) files"
phase "0 input folders"

if [ -n "$REVIEWED" ]; then
  stamp "=== 1. reviews -> labels ($PREV -> $NEW)"
  mkdir -p rev_in
  $PY src/reviewed_to_labels.py --reviewed "$REVIEWED" --tiles-dir $TILES_ALL --out-dir rev_in/${TAG}_out > rev_in/$TAG.log 2>&1 \
    || { stamp "reviewed_to_labels FAILED"; tail -8 rev_in/$TAG.log; exit 1; }
  tail -12 rev_in/$TAG.log
  $PY src/merge_review_outputs.py --base-dir $OUT --suffix $PREV --out-suffix $NEW --new rev_in/${TAG}_out \
    --eval-labels $OUT/labels_all2_for_eval.gpkg --tiles-out $OUT/reviewed_tiles_$NEW.txt || { stamp "merge FAILED"; exit 1; }
  phase "1 reviews -> labels"
fi

stamp "=== 2. split quarry tiles: reviewed / rest"
rm -rf $Q/imagery_reviewed $Q/imagery_rest; mkdir -p $Q/imagery_reviewed $Q/imagery_rest
for f in $TILES_ALL/*.tif; do
  id=$(basename $f .tif)
  if grep -qx "$id" $OUT/reviewed_tiles_$LABELS.txt; then ln -s "$(realpath $f)" $Q/imagery_reviewed/; else ln -s "$(realpath $f)" $Q/imagery_rest/; fi
done
stamp "reviewed tiles: $(ls $Q/imagery_reviewed | wc -l), rest: $(ls $Q/imagery_rest | wc -l)"
phase "2 tile split"

stamp "=== 3. retrain + rescore (tile cache: ${CACHE_FLAG:-off})"
rm -rf $RUN; mkdir $RUN
for f in feature_cache.pkl polygons_cache.gpkg; do ln -s "$(realpath $BASE/$f)" $RUN/$f; done
$PY src/otr_obia_pipeline.py --imagery-dir data/imagery --labels-path data/dump_labels.gpkg --mine-boundary-path data/mine_boundaries.gpkg \
  --s1-dir data/all_s1 --dem-dir data/all_dem --s2my-dir data/s2_multiyear_all --s2my-bands $BANDS --target-segment-px 12 \
  --max-segments-per-mine 1000000 --n-jobs 12 --use-cache $CACHE_FLAG --labeled-imagery-dir data/run15/labeled --extra-labels-path data/run15/labels_extra.gpkg \
  --reviewed-imagery-dir $Q/imagery_reviewed --extra-imagery-dir $Q/imagery_rest \
  --add-labels-path $OUT/labels_reviewed_dump_$LABELS.gpkg --ignore-points-path data/run15/ignore_points.geojson,$OUT/ignore_reviewed_$LABELS.geojson \
  --hard-negatives-path $OUT/negatives_${LABELS}_clean.gpkg --hard-negative-factor 20 \
  --osm-buildings-dir $Q/osm_buildings --osm-roads-dir $Q/osm_roads --osm-poi-dir $Q/osm_poi --output-dir $RUN > log_q$TAG.txt 2>&1 \
  || { stamp "pipeline FAILED"; tail -15 log_q$TAG.txt; exit 1; }
# split the pipeline time into its phases from the log time stamps
ts() { grep -m1 -E "^[0-9:]+ \[INFO\] $1" log_q$TAG.txt | cut -c1-8; }
sec() { local h m s; IFS=: read h m s <<< "$1"; echo $((10#$h*3600+10#$m*60+10#$s)); }
A=$(ts "1\) Lade"); B=$(ts "2\) Training"); C=$(ts "3\) Vollpr"); D=$(grep -m1 "road_density_150m berechnet" log_q$TAG.txt | cut -c1-8); E=$(grep -m1 "Exportiert nach" log_q$TAG.txt | cut -c1-8)
if [ -n "$A" ] && [ -n "$B" ] && [ -n "$C" ] && [ -n "$D" ] && [ -n "$E" ]; then
  printf "%-48s %4d s\n" "3a tile features (segmentation, cache)" $(( $(sec $B) - $(sec $A) )) | tee -a timing_$TAG.txt
  printf "%-48s %4d s\n" "3b training + cross-validation" $(( $(sec $C) - $(sec $B) )) | tee -a timing_$TAG.txt
  printf "%-48s %4d s\n" "3c scoring + OSM flags + road density" $(( $(sec $D) - $(sec $C) )) | tee -a timing_$TAG.txt
  printf "%-48s %4d s\n" "3d writing segments_classified.gpkg" $(( $(sec $E) - $(sec $D) )) | tee -a timing_$TAG.txt
fi
phase "3 pipeline (total)"
grep -c "aus dem Zwischenspeicher geladen" log_q$TAG.txt | sed 's/^/tiles loaded from the tile cache: /' | tee -a timing_$TAG.txt

stamp "=== 4. candidate lists (top 2% area per mine, rank by size x mean score, with context)"
COMMONC="--segments $RUN/segments_classified.gpkg --preselect --area-pct 2 --labels $OUT/labels_${LABELS}_for_eval.gpkg --exclude-known \
  --rank-by score --min-area-m2 2500 --max-site-m2 40000 --max-urban-frac 0.3 --exclude-areas $OUT/negatives_${LABELS}_clean.gpkg,$OUT/ignore_reviewed_$LABELS.geojson \
  --operations data/mine_boundaries.gpkg,data/northChileQuarrys.geojson${OPS_EXTRA:+,$OPS_EXTRA}"
$PY src/candidate_sites.py $COMMONC --top 2500 --sample-every 5 --out $RUN/candidate_sites_${TAG}_sample5.gpkg > log_${TAG}_sample5.txt 2>&1
$PY src/candidate_sites.py $COMMONC --top 500 --out $RUN/candidate_sites_${TAG}_top500.gpkg > log_${TAG}_top500.txt 2>&1
phase "4 candidate lists"

stamp "=== 5. check on the 138 known mines"
{
  if [ -n "$COMPARE" ]; then
    $PY src/eval_segment_size.py --labels $OUT/labels_all2_for_eval.gpkg --run before=$COMPARE --run $TAG=$RUN
    $PY src/eval_false_alarms.py --negatives $OUT/negatives_all_clean.gpkg --run before=$COMPARE --run $TAG=$RUN
  else
    $PY src/eval_segment_size.py --labels $OUT/labels_all2_for_eval.gpkg --run $TAG=$RUN
    $PY src/eval_false_alarms.py --negatives $OUT/negatives_all_clean.gpkg --run $TAG=$RUN
  fi
} > results_$TAG.txt 2>&1
phase "5 evaluations"
printf "%-48s %4d s\n" "TOTAL" $(( $(date +%s) - T0 )) | tee -a timing_$TAG.txt
stamp "=== ALL DONE ($TAG) ==="
