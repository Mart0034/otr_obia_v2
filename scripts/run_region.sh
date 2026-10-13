#!/bin/bash
# Look for dumps in a whole region (also far from known mines/quarries) with the model trained on the reviewed labels.
#
#   bash scripts/run_region.sh TAG --bounds W,S,E,N --labels all7 [--tile-m 2700] [--exclude-near FILE] [--exclude-buffer-m 1500]
#                              [--limit N] [--ops-extra FILE] [--no-prescreen]
#
# Steps: region grid -> drop tiles near known mines (optional) and tiles that cannot hold a dump (water, snow, forest, town;
# ESA WorldCover, a few kB per tile) -> download Sentinel-2, Sentinel-1, elevation, multi-year brightness for all remaining tiles
# at the same time, OSM cut from the local Chile file -> score with the SAVED model (training only if the labels changed)
# -> top-500 candidate list. Only the new tiles are scored and exported.
# Output: data/regions/TAG/ (tiles), out_px12_qTAG/ (segments, candidate_sites_TAG_top500.gpkg), timing_TAG.txt
cd /home/container || exit 1
export LD_LIBRARY_PATH=/home/container/libexpat_extracted/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PY=venv/bin/python3; OUT=data/reviewed/out; Q=data/sites_quarries
TAG=$1; shift
BOUNDS=""; LABELS=""; TILE_M=2700; EXCL=""; EXCL_BUF=1500; LIMIT=""; OPS_EXTRA=""; PRESCREEN="--prescreen"; OSM_BUILD=data/osm_chile
while [ $# -gt 0 ]; do
  case $1 in
    --bounds) BOUNDS=$2; shift 2;; --labels) LABELS=$2; shift 2;; --tile-m) TILE_M=$2; shift 2;;
    --exclude-near) EXCL=$2; shift 2;; --exclude-buffer-m) EXCL_BUF=$2; shift 2;; --limit) LIMIT="--limit $2"; shift 2;;
    --ops-extra) OPS_EXTRA=$2; shift 2;; --no-prescreen) PRESCREEN=""; shift;; --osm-build-dir) OSM_BUILD=$2; shift 2;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -z "$TAG" ] || [ -z "$BOUNDS" ] || [ -z "$LABELS" ] && { echo "usage: run_region.sh TAG --bounds W,S,E,N --labels SUFFIX [...]"; exit 2; }
R=data/regions/$TAG; RUN=out_px12_q$TAG
T0=$(date +%s); T1=$T0; : > timing_$TAG.txt
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
phase() { printf "%-44s %5d s (total %d s)\n" "$1" $(( $(date +%s)-T1 )) $(( $(date +%s)-T0 )) | tee -a timing_$TAG.txt; T1=$(date +%s); }
set -o pipefail

stamp "=== 1. plan + land-cover pre-screen + parallel download"
EXARGS=""; [ -n "$EXCL" ] && EXARGS="--exclude-near $EXCL --exclude-buffer-m $EXCL_BUF"
OSMARGS=""; [ -d "$OSM_BUILD" ] && OSMARGS="--osm-build-dir $OSM_BUILD"
$PY src/sites_from_vector.py --grid-bounds=$BOUNDS --grid-tile-m $TILE_M $PRESCREEN $EXARGS --prefix $TAG \
  --out-dir $R --fetch $LIMIT --n-workers 12 $OSMARGS > log_${TAG}_fetch.txt 2>&1 || { stamp "download FAILED"; tail -10 log_${TAG}_fetch.txt; exit 1; }
grep -E "Kacheln geplant|übersprungen|Vorauswahl|Fertig in" log_${TAG}_fetch.txt | cut -c1-200
phase "1 plan + download"

stamp "=== 2. keep tiles with all inputs, link into the shared folders"
rm -rf $R/complete; mkdir -p $R/complete/imagery; n=0
for f in $R/imagery/*.tif; do id=$(basename $f .tif)
  if [ -e $R/sentinel1/$id.tif ] && [ -e $R/dem/$id.tif ] && [ -e $R/s2my/$id.tif ]; then
    ln -sf "$(realpath $f)" $R/complete/imagery/$id.tif
    ln -sf "$(realpath $R/sentinel1/$id.tif)" data/all_s1/$id.tif
    ln -sf "$(realpath $R/dem/$id.tif)" data/all_dem/$id.tif
    ln -sf "$(realpath $R/s2my/$id.tif)" data/s2_multiyear_all/$id.tif
    for k in roads buildings poi; do [ -e $R/osm_$k/$id.gpkg ] && ln -sf "$(realpath $R/osm_$k/$id.gpkg)" $Q/osm_$k/$id.gpkg; done
    n=$((n+1))
  fi; done
stamp "complete tiles: $n of $(ls $R/imagery/*.tif 2>/dev/null | wc -l)"
[ "$n" -eq 0 ] && { stamp "no complete tiles"; exit 1; }
phase "2 assemble"

stamp "=== 3. score with the model for $LABELS (saved model if the labels are unchanged)"
rm -rf $RUN; mkdir $RUN
for f in feature_cache.pkl polygons_cache.gpkg; do ln -s "$(realpath out_px12_my_bright/$f)" $RUN/$f; done
$PY src/otr_obia_pipeline.py --imagery-dir data/imagery --labels-path data/dump_labels.gpkg --mine-boundary-path data/mine_boundaries.gpkg \
  --s1-dir data/all_s1 --dem-dir data/all_dem --s2my-dir data/s2_multiyear_all --s2my-bands s2my_bright_early,s2my_bright_late --target-segment-px 12 \
  --max-segments-per-mine 1000000 --n-jobs 12 --use-cache --tile-cache-dir data/tile_cache --model-cache-dir data/model_cache \
  --labeled-imagery-dir data/run15/labeled --extra-labels-path data/run15/labels_extra.gpkg \
  --reviewed-imagery-dir $Q/imagery_reviewed --extra-imagery-dir $R/complete/imagery --export-extra-only --skip-polygons-extra \
  --add-labels-path $OUT/labels_reviewed_dump_$LABELS.gpkg --ignore-points-path data/run15/ignore_points.geojson,$OUT/ignore_reviewed_$LABELS.geojson \
  --hard-negatives-path $OUT/negatives_${LABELS}_clean.gpkg --hard-negative-factor 20 \
  --osm-buildings-dir $Q/osm_buildings --osm-roads-dir $Q/osm_roads --osm-poi-dir $Q/osm_poi --output-dir $RUN > log_q$TAG.txt 2>&1 \
  || { stamp "pipeline FAILED"; tail -15 log_q$TAG.txt; exit 1; }
grep -E "Modell-Zwischenspeicher" log_q$TAG.txt | cut -c1-160
phase "3 scoring"

stamp "=== 4. candidate list"
$PY src/candidate_sites.py --segments $RUN/segments_classified.gpkg --preselect --area-pct 2 --labels $OUT/labels_${LABELS}_for_eval.gpkg --exclude-known \
  --rank-by score --min-area-m2 2500 --max-site-m2 40000 --max-urban-frac 0.3 --exclude-areas $OUT/negatives_${LABELS}_clean.gpkg,$OUT/ignore_reviewed_$LABELS.geojson \
  --operations data/mine_boundaries.gpkg,data/northChileQuarrys.geojson${OPS_EXTRA:+,$OPS_EXTRA} --top 500 --out $RUN/candidate_sites_${TAG}_top500.gpkg > log_${TAG}_top500.txt 2>&1
phase "4 candidate list"
printf "%-44s %5d s\n" "TOTAL" $(( $(date +%s)-T0 )) | tee -a timing_$TAG.txt
stamp "=== ALL DONE ($TAG) ==="
