#!/bin/bash
# Pilot run in a NEW region with the model trained on Chile (no retraining labels from the new region).
#   bash scripts/run_pilot.sh NAME --crs EPSG:32612 --n-per 150 --region COUNTRY W E S N [--region COUNTRY W E S N ...] [--labels all8]
# Example (US southwest + northern Mexico, UTM zone 12N):
#   bash scripts/run_pilot.sh usmx --crs EPSG:32612 --n-per 150 --region "United States of America" -121 -103 31 42 --region Mexico -118 -97 22 33
#
# Steps: places from the global mining polygons per region (0.3-15 km2, a random sample of --n-per per region) ->
# download the tiles (Sentinel-2/-1, DEM, multi-year; no local OSM, so the urban filter is off) -> score with the cached
# Chile model, exporting ONLY the new tiles (one coordinate system) -> candidate lists with Google Maps links.
# Run on the VPS in /home/container. All tiles of one run share one metric CRS (--crs): pick the UTM zone of the region.
cd /home/container || exit 1
export LD_LIBRARY_PATH=/home/container/libexpat_extracted/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PY=venv/bin/python3; OUT=data/reviewed/out; Q=data/sites_quarries; BASE=out_px12_my_bright; BANDS=s2my_bright_early,s2my_bright_late
NAME=$1; shift; CRS=EPSG:32612; NPER=150; LABELS=all8; REGIONS=()
while [ $# -gt 0 ]; do
  case $1 in
    --crs) CRS=$2; shift 2;; --n-per) NPER=$2; shift 2;; --labels) LABELS=$2; shift 2;;
    --region) REGIONS+=("$2|$3|$4|$5|$6"); shift 6;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -z "$NAME" ] || [ ${#REGIONS[@]} -eq 0 ] && { echo "usage: run_pilot.sh NAME --crs CRS --n-per N --region COUNTRY W E S N ..."; exit 2; }
R=data/pilot_$NAME; RUN=out_pilot_$NAME; mkdir -p $R
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
SHP=$(ls data/chile_new/tang/*.shp | head -1)

stamp "=== 1. places per region"
i=0; for r in "${REGIONS[@]}"; do IFS='|' read C W E S N <<< "$r"; i=$((i+1))
  $PY src/region_vector.py --polys "$SHP" --country "$C" --work-crs $CRS --lon-range $W $E --lat-range $S $N \
    --min-area-m2 300000 --out $R/places_$i.gpkg --ops-out $R/ops_$i.gpkg > $R/log_places_$i.txt 2>&1 || { stamp "places FAILED ($C)"; tail -8 $R/log_places_$i.txt; exit 1; }
  grep -E "Polygone|Orte" $R/log_places_$i.txt | cut -c1-150
done
$PY - <<P
import glob, geopandas as gpd, pandas as pd
parts, ops = [], []
for f in sorted(glob.glob("$R/places_*.gpkg")):
    g = gpd.read_file(f); g = g[(g["area_m2"] >= 3e5) & (g["area_m2"] <= 15e6)]      # no giant evaporation ponds
    parts.append(g.sample(min(len(g), $NPER), random_state=7))
pd.concat(parts, ignore_index=True).pipe(lambda d: gpd.GeoDataFrame(d, crs=parts[0].crs)).to_file("$R/places.gpkg", driver="GPKG")
gpd.GeoDataFrame(pd.concat([gpd.read_file(f)[["geometry"]] for f in sorted(glob.glob("$R/ops_*.gpkg"))], ignore_index=True), crs=parts[0].crs).to_file("$R/ops_all.gpkg", driver="GPKG")
print(sum(len(p) for p in parts), "places sampled")
P

stamp "=== 2. download the tiles"
$PY src/sites_from_vector.py --input $R/places.gpkg --work-crs $CRS --buffer-m 500 --merge-overlaps --max-tile-m 6000 --prefix px \
  --prescreen --out-dir $R/tiles --fetch --n-workers 12 > $R/log_fetch.txt 2>&1 || { stamp "download FAILED"; tail -10 $R/log_fetch.txt; exit 1; }
grep -E "Kacheln geplant|übersprungen|Fertig in|unvollständig" $R/log_fetch.txt | cut -c1-180

stamp "=== 3. link tiles that have all inputs"
mkdir -p data/all_s1 data/all_dem data/s2_multiyear_all $R/ok_imagery
n=0; for f in $R/tiles/imagery/*.tif; do id=$(basename $f .tif)
  if [ -e $R/tiles/sentinel1/$id.tif ] && [ -e $R/tiles/dem/$id.tif ] && [ -e $R/tiles/s2my/$id.tif ]; then
    ln -sf "$(realpath $f)" $R/ok_imagery/$id.tif
    ln -sf "$(realpath $R/tiles/sentinel1/$id.tif)" data/all_s1/$id.tif; ln -sf "$(realpath $R/tiles/dem/$id.tif)" data/all_dem/$id.tif
    ln -sf "$(realpath $R/tiles/s2my/$id.tif)" data/s2_multiyear_all/$id.tif; n=$((n+1))
  fi; done
stamp "tiles with all inputs: $n"

stamp "=== 4. score (cached Chile model, export only the new tiles)"
rm -rf $RUN; mkdir $RUN; for f in feature_cache.pkl polygons_cache.gpkg; do ln -s "$(realpath $BASE/$f)" $RUN/$f; done
$PY src/otr_obia_pipeline.py --imagery-dir data/imagery --labels-path data/dump_labels.gpkg --mine-boundary-path data/mine_boundaries.gpkg \
  --s1-dir data/all_s1 --dem-dir data/all_dem --s2my-dir data/s2_multiyear_all --s2my-bands $BANDS --target-segment-px 12 \
  --max-segments-per-mine 1000000 --n-jobs 12 --use-cache --tile-cache-dir data/tile_cache --model-cache-dir data/model_cache \
  --labeled-imagery-dir data/run15/labeled --extra-labels-path data/run15/labels_extra.gpkg \
  --reviewed-imagery-dir $Q/imagery_reviewed --extra-imagery-dir $R/ok_imagery --export-extra-only --skip-polygons-extra \
  --add-labels-path $OUT/labels_reviewed_dump_$LABELS.gpkg --ignore-points-path data/run15/ignore_points.geojson,$OUT/ignore_reviewed_$LABELS.geojson \
  --hard-negatives-path $OUT/negatives_${LABELS}_clean.gpkg --hard-negative-factor 20 --output-dir $RUN > log_pilot_$NAME.txt 2>&1 \
  || { stamp "pipeline FAILED"; tail -15 log_pilot_$NAME.txt; exit 1; }

stamp "=== 5. candidate lists with Google Maps links"
COMMONC="--segments $RUN/segments_classified.gpkg --preselect --area-pct 2 --rank-by mean_proba --min-area-m2 2500 --max-site-m2 40000 --operations $R/ops_all.gpkg"
$PY src/candidate_sites.py $COMMONC --top 300 --out $RUN/candidate_sites_${NAME}_top300.gpkg > log_pilot_${NAME}_top300.txt 2>&1
$PY src/candidate_sites.py $COMMONC --top 1000 --out $RUN/candidate_sites_${NAME}_top1000.gpkg > log_pilot_${NAME}_top1000.txt 2>&1
stamp "=== PILOT DONE ($NAME): $RUN/candidate_sites_${NAME}_top300.gpkg (+ .csv with maps_url) ==="
