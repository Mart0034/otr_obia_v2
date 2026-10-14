#!/bin/bash
# Overnight run: look beyond mines. Landfills, recycling yards, tyre shops, works and industrial land in northern Chile
# (from the local OSM file), not yet covered by any tile. Uses the cached Chile model, exports only the new tiles.
#   bash scripts/run_industrial.sh [NAME=ind1] [LIMIT_TILES=1200] [LABELS=all8]
# Output: out_<NAME>/candidate_sites_<NAME>_top500.gpkg / _top1500.gpkg (+ .csv with maps_url), timing_<NAME>.txt
cd /home/container || exit 1
export LD_LIBRARY_PATH=/home/container/libexpat_extracted/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PY=venv/bin/python3; OUT=data/reviewed/out; Q=data/sites_quarries; BASE=out_px12_my_bright; BANDS=s2my_bright_early,s2my_bright_late
NAME=${1:-ind1}; LIMIT=${2:-1200}; LABELS=${3:-all8}
R=data/$NAME; RUN=out_$NAME; mkdir -p $R; T0=$(date +%s); : > timing_$NAME.txt
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
mark() { printf "%-34s %5d s (total %d s)\n" "$1" $(( $(date +%s)-T1 )) $(( $(date +%s)-T0 )) | tee -a timing_$NAME.txt; T1=$(date +%s); }
T1=$T0

stamp "=== 1. places from OSM"
$PY src/osm_places.py --pbf data/osm/chile-latest.osm.pbf --south -33 --north -17.5 --limit 1800 \
  --exclude-tiles-dir data/imagery --exclude-tiles-dir data/sites_quarries/imagery --exclude-tiles-dir data/chile_all/imagery \
  --exclude-tiles-dir data/run15/labeled --out $R/places.gpkg > $R/log_places.txt 2>&1 || { stamp "places FAILED"; tail -8 $R/log_places.txt; exit 1; }
tail -4 $R/log_places.txt | cut -c1-200
mark "1 places"

stamp "=== 2. download tiles (no land-cover pre-screen: industrial land counts as built-up)"
$PY src/sites_from_vector.py --input $R/places.gpkg --buffer-m 400 --merge-overlaps --max-tile-m 6000 --prefix in \
  --out-dir $R/tiles --fetch --limit $LIMIT --n-workers 12 --osm-build-dir data/osm_chile > $R/log_fetch.txt 2>&1 || { stamp "download FAILED"; tail -10 $R/log_fetch.txt; exit 1; }
grep -E "Kacheln geplant|übersprungen|Fertig in|unvollständig" $R/log_fetch.txt | cut -c1-200
mark "2 download"

stamp "=== 3. link tiles that have all inputs"
mkdir -p data/all_s1 data/all_dem data/s2_multiyear_all $R/ok_imagery
n=0; for f in $R/tiles/imagery/*.tif; do id=$(basename $f .tif)
  if [ -e $R/tiles/sentinel1/$id.tif ] && [ -e $R/tiles/dem/$id.tif ] && [ -e $R/tiles/s2my/$id.tif ]; then
    ln -sf "$(realpath $f)" $R/ok_imagery/$id.tif; ln -sf "$(realpath $R/tiles/sentinel1/$id.tif)" data/all_s1/$id.tif
    ln -sf "$(realpath $R/tiles/dem/$id.tif)" data/all_dem/$id.tif; ln -sf "$(realpath $R/tiles/s2my/$id.tif)" data/s2_multiyear_all/$id.tif; n=$((n+1))
  fi; done
stamp "tiles with all inputs: $n"
mark "3 link"

stamp "=== 4. score with the cached Chile model (only the new tiles)"
rm -rf $RUN; mkdir $RUN; for f in feature_cache.pkl polygons_cache.gpkg; do ln -s "$(realpath $BASE/$f)" $RUN/$f; done
$PY src/otr_obia_pipeline.py --imagery-dir data/imagery --labels-path data/dump_labels.gpkg --mine-boundary-path data/mine_boundaries.gpkg \
  --s1-dir data/all_s1 --dem-dir data/all_dem --s2my-dir data/s2_multiyear_all --s2my-bands $BANDS --target-segment-px 12 \
  --max-segments-per-mine 1000000 --n-jobs 12 --use-cache --tile-cache-dir data/tile_cache --model-cache-dir data/model_cache \
  --labeled-imagery-dir data/run15/labeled --extra-labels-path data/run15/labels_extra.gpkg \
  --reviewed-imagery-dir $Q/imagery_reviewed --extra-imagery-dir $R/ok_imagery --export-extra-only --skip-polygons-extra \
  --add-labels-path $OUT/labels_reviewed_dump_$LABELS.gpkg --ignore-points-path data/run15/ignore_points.geojson,$OUT/ignore_reviewed_$LABELS.geojson \
  --hard-negatives-path $OUT/negatives_${LABELS}_clean.gpkg --hard-negative-factor 20 \
  --osm-buildings-dir $R/tiles/osm_buildings --osm-roads-dir $R/tiles/osm_roads --osm-poi-dir $R/tiles/osm_poi \
  --output-dir $RUN > log_$NAME.txt 2>&1 || { stamp "pipeline FAILED"; tail -15 log_$NAME.txt; exit 1; }
mark "4 score"

stamp "=== 5. candidate lists with Google Maps links"
COMMONC="--segments $RUN/segments_classified.gpkg --preselect --area-pct 2 --labels $OUT/labels_${LABELS}_for_eval.gpkg --exclude-known \
  --rank-by mean_proba --min-area-m2 1500 --max-site-m2 40000 --max-urban-frac 0.5 \
  --exclude-areas $OUT/negatives_${LABELS}_clean.gpkg,$OUT/ignore_reviewed_$LABELS.geojson \
  --operations $R/places.gpkg,data/mine_boundaries.gpkg,data/northChileQuarrys.geojson"
$PY src/candidate_sites.py $COMMONC --top 500 --out $RUN/candidate_sites_${NAME}_top500.gpkg > log_${NAME}_top500.txt 2>&1
$PY src/candidate_sites.py $COMMONC --top 1500 --out $RUN/candidate_sites_${NAME}_top1500.gpkg > log_${NAME}_top1500.txt 2>&1
mark "5 candidate lists"
stamp "=== INDUSTRIAL RUN DONE ($NAME) ==="
