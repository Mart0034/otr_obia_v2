*Read this in: **English** | [Deutsch](README.de.md)*

# OTR OBIA Pipeline

Detects old tire dump sites ("OTR" stands for Old Tire Dump) at mining
sites from Sentinel-2 satellite imagery, using an object-based approach
(OBIA): group each image into homogeneous patches ("segments") first,
then classify each patch as a whole with a Random Forest, instead of
classifying every single pixel with a CNN. See `src/otr_obia_pipeline.py`
for the full background and reasoning in the module docstring.

## Status

- Pipeline code: implemented and bug-fixed.
- Automated tests: in place (`tests/`), run on synthetic/made-up data.
- Real data: **not yet available**. This has only been run against
  synthetic test fixtures so far, never against real Sentinel-2 imagery.
  See "Data you need to provide" below.

## Setup

Requires Python 3.10+. A dedicated virtual environment is recommended,
since some of these libraries (rasterio, geopandas) can conflict with
the Python bundled inside QGIS, so don't reuse the QGIS Python
environment.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # for just running the pipeline
# or, to also run the tests:
pip install -r requirements-dev.txt
```

## Data you need to provide

The pipeline does not come with any data. You need to place two things
under `data/` (this folder is git-ignored, so nothing here ever gets
committed):

1. **`data/imagery/*.tif`**: one GeoTIFF per mine, each with exactly
   10 bands, in this order:

   ```
   blue, green, red, nir, swir1, swir2, ndvi, ndwi, dsi, bsi
   ```

   This is expected to already exist from "Stage 2" of the earlier
   feasibility-study project. Ask whoever ran that stage where the
   finished per-mine stacks live.

2. **`data/dump_labels.gpkg`**: a GeoPackage (or Shapefile) with the
   digitized tire-dump polygons (drawn in QGIS), with a column (default
   name: `mine_id`) that identifies which mine each polygon belongs to.

3. **Make sure mine IDs match.** The pipeline guesses a mine's ID from
   its image filename (e.g. `mine_12.tif` becomes `mine_12`, `7.tif`
   becomes `7`). That guessed ID has to line up with the values in the
   labels file's `mine_id` column, or that mine's segments will silently
   all be labeled "no dump". You'll get a clear warning in the log if
   this happens, but it's worth double-checking the naming scheme up
   front (see `CONFIG["mine_id_field"]` / `extract_mine_id()` in the
   script).

4. **`data/mine_boundaries.gpkg`** (optional but recommended): the real
   mine boundary polygons (not the buffered image extent). Each Sentinel-2
   export includes a 500m buffer of surrounding terrain around the mine,
   which can never contain a tire dump but still gets segmented and
   classified. Pointing `--mine-boundary-path data/mine_boundaries.gpkg`
   (column `mine_id` by default, see `--mine-boundary-id-field`) at this
   file drops every segment that falls entirely outside the real
   boundary before classification, removing a meaningful share of false
   positives for free. Without this option the pipeline behaves exactly
   as before (no filtering).

## Checking your data before running

Before running the full pipeline (which can take a while), run the data
checker. It reports, in seconds, whether your GeoTIFFs have the right
number of bands, how many pixels are missing/no-data, and whether every
mine's ID actually matches an entry in the labels file:

```bash
python3 src/check_data.py --imagery-dir data/imagery --labels-path data/dump_labels.gpkg
```

It exits with a non-zero status if anything looks wrong, so a mine
listed as "KEIN LABEL-MATCH" (no matching label) means that mine's ID
in the imagery filename doesn't line up with the labels file. Fix that
before running the full pipeline, not after.

## Running the pipeline

You no longer need to edit the code to change settings. Either edit the
`CONFIG` dictionary at the top of `src/otr_obia_pipeline.py`, or pass
options on the command line:

```bash
python3 src/otr_obia_pipeline.py \
  --imagery-dir data/imagery \
  --labels-path data/dump_labels.gpkg \
  --output-dir output
```

Run `python3 src/otr_obia_pipeline.py --help` to see every available
option (same options work for `check_data.py`). You can also collect a
whole set of settings in a JSON file and pass it with `--config`:

```json
{"imagery_dir": "data/imagery", "min_overlap_ratio": 0.4, "n_estimators": 600}
```

```bash
python3 src/otr_obia_pipeline.py --config my_settings.json
```

Individual flags always win over the config file, which always wins
over the built-in defaults in `CONFIG`.

Running the pipeline (with either method) will:
1. Build the segment dataset from every mine's imagery + labels.
2. Train a Random Forest and cross-validate it (grouped by mine, so no
   mine's data leaks between training and validation).
3. Classify every segment and export `output/segments_classified.gpkg`.

Load that GeoPackage into QGIS (drag & drop) and color by the
`dump_proba` column (Graduated, threshold ~0.5 by default, see below) to
inspect results.

Everything the pipeline does is logged with timestamps to the console,
including warnings for anything that looks off (mismatched mine IDs,
no-data pixels found, a cross-validation fold with only one class,
etc.). Check the log output for warnings after a run.

### Choosing a classification threshold

`--classification-threshold` (default `0.5`) controls the cutoff on
`dump_proba` used both for the cross-validation metrics and for the
`dump_pred` column in the exported map. With data this imbalanced, 0.5
is not necessarily the best trade-off — a higher threshold (e.g. `0.7`
or `0.8`) sharply cuts the number of false-positive segments at the cost
of missing a few of the fainter true dumps. Every run now logs a
threshold-sweep table (out-of-fold, so it's an honest comparison) showing
how many mines get detected and how many segments get flagged positive
at several thresholds, so you can pick a value and re-run with
`--use-cache` (skips re-segmenting, only retrains/re-exports) instead of
guessing.

### Requiring a neighboring segment for medium-confidence detections (optional)

`--require-neighbor-below` (default off) drops a positive segment whose
`dump_proba` is below the given value unless a spatially adjacent segment
in the same mine was also predicted positive. It targets a specific,
recurring pattern in the false positives: isolated single segments of
middling confidence strung along roads and pit edges. Segments at or above
the given value are never dropped, even with no neighbor, some real dumps
are exactly one segment (mine_012's only known dump is a single
high-confidence segment with nothing next to it), so a blanket "needs at
least 2 segments" rule would have deleted that hit. A reasonable starting
value is `0.6`, the same threshold the sweep table already reports on.

### Requiring a compact cluster shape for medium-confidence detections (optional)

`--require-compact-cluster-below` (default off) drops a positive segment
whose `dump_proba` is below the given value unless the connected cluster
of touching positive segments it belongs to is reasonably blob-shaped
rather than a long thin chain. It's a different check from
`--require-neighbor-below`, not a replacement: a chain of segments strung
along a road or a cliff edge all touch each other, so every segment in it
already has a supporting neighbor and sails straight through that filter.
What's wrong with a chain isn't isolation, it's shape. Cluster shape is
measured the same way as `shape_compactness` (4π·area/perimeter²),
applied to the whole connected cluster rather than one segment, which
stays accurate for a winding road, unlike a bounding-box aspect-ratio
test, which a curved chain can dodge by curling back into a squarer
bounding box. `--min-cluster-compactness` (default `0.15`) sets the
cutoff; calibrated against the real data, where multi-segment
false-positive clusters had a median compactness of 0.27 (20% fell below
0.15) against 0.34 for real multi-segment dump clusters (none fell below
0.15). Segments at or above the confidence value are never dropped,
same reasoning as the neighbor filter.

### Adding Sentinel-1 radar features (optional)

Radar measures surface roughness rather than color: a pile of tires
scatters the signal strongly, a graded haul road hardly at all. That is
the confusion behind most optical false positives. It also isn't affected
by cloud cover.

1. Download radar data matching each mine image (free, no account, from
   Microsoft Planetary Computer):

   ```bash
   python src/fetch_sentinel1.py --imagery-dir data/imagery --out-dir data/sentinel1 \
     --start 2023-01-01 --end 2023-12-31
   ```

   For each mine this takes the median of up to 12 radar scenes from the
   period (`--max-scenes`), which suppresses radar noise ("speckle"), and
   resamples it onto exactly the same pixel grid as that mine's Sentinel-2
   image. Already downloaded mines are skipped, so an interrupted run can
   simply be restarted. Pick the same period as the Sentinel-2 composite if
   you know it.

2. Run the pipeline with `--s1-dir data/sentinel1`. Segmentation still uses
   only Sentinel-2; radar adds six features per segment (mean and variation
   of VV, VH and the VH/VV ratio). A mine without a radar file gets a
   warning and empty radar features. If `--use-cache` finds an old cache
   without radar features, the pipeline rebuilds it.

### Adding terrain features from the Copernicus DEM (optional)

Shadowed mountainsides look dark and patchy in the optical bands, much like
a tire dump. Terrain features let the model tell a steep, shaded slope from
the flat ground dumps are usually on.

1. Download the elevation model (Copernicus GLO-30, 30m, free, no account)
   onto each mine's pixel grid:

   ```bash
   python src/fetch_dem.py --imagery-dir data/imagery --out-dir data/dem
   ```

2. Run the pipeline with `--dem-dir data/dem` (combines with `--s1-dir`).
   Per segment this adds the mean and variation of:
   - `dem_slope`: slope in degrees
   - `dem_tpi`: elevation minus the average of the surrounding ~300m
     (`dem_tpi_window_px`); positive on ridges, negative in valleys
   - `dem_northness`: -1..1, positive on north-facing slopes. In the
     southern hemisphere the sun is to the north, so south-facing slopes
     (negative) are the shaded ones. Flat ground is 0.
   - `dem_hillshade_dec` / `dem_hillshade_jun`: expected illumination
     (-1..1, roughly) from the real sun position at local midday on the
     southern-hemisphere summer and winter solstices, using an actual
     solar-position calculation rather than a plain north/south proxy.
     More precise than `dem_northness` in a narrow valley, where a segment
     can straddle both a sunlit and a shadowed wall and the northness
     *average* washes out even though half the segment is genuinely dark.
     It only models a surface's own orientation toward the sun, not
     shadows cast by neighboring terrain (that would need ray-tracing
     across the whole DEM).

   Absolute elevation is deliberately left out: it ranges from the coast to
   ~4000m between mines and would mostly tell the model which mine it is
   looking at.

### Adding multi-date Sentinel-2 features (optional)

Shadows move with the sun, tire dumps don't. Over Antofagasta the sun is
~23° from vertical at the Sentinel-2 overpass in December and ~56° in June,
so a slope or pit edge that's in shadow in winter is sunlit in summer, while
a dump stays about equally dark all year.

1. Compute per-pixel brightness statistics from up to 8 low-cloud Sentinel-2
   scenes spread over a year (free, no account):

   ```bash
   python src/fetch_s2_timeseries.py --imagery-dir data/imagery --out-dir data/s2_temporal \
     --start 2023-01-01 --end 2023-12-31
   ```

   Clouds and cloud shadows are masked out using the scene classification
   layer; terrain shadow is deliberately kept. Use a full year so both the
   summer and winter sun positions are included.

2. Run the pipeline with `--s2t-dir data/s2_temporal` (combines with
   `--s1-dir` and `--dem-dir`). Per segment this adds the mean and
   variation of:
   - `s2t_bright_cv`: how much brightness varies over the year (0 = constant)
   - `s2t_bright_min_ratio`: darkest scene divided by the typical (median)
     brightness; low means the spot is sometimes heavily shadowed

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

The tests use small made-up "mine" images and labels generated on the
fly (see `conftest.py`). They don't need any real data, and should run
in a few seconds. They also run automatically on every push via GitHub
Actions (`.github/workflows/tests.yml`).

## Known limitations

- GLCM texture features are computed on each segment's bounding box,
  not its exact polygon shape. This is a simple approximation, not an
  exact calculation.
- Model settings (`n_segments_per_mine`, `compactness`,
  `min_overlap_ratio`, `n_estimators`) are initial defaults and have not
  been tuned against real data yet.
- With very few mines, cross-validation folds can end up with only one
  class present. The pipeline handles this without crashing, but the
  resulting fold score isn't very informative. Check the logged
  warnings for which mines this affected.
