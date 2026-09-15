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

## Running the pipeline

Edit the `CONFIG` dictionary at the top of `src/otr_obia_pipeline.py` to
point at your data (paths, band names, thresholds), then run:

```bash
python3 src/otr_obia_pipeline.py
```

This will:
1. Build the segment dataset from every mine's imagery + labels.
2. Train a Random Forest and cross-validate it (grouped by mine, so no
   mine's data leaks between training and validation).
3. Classify every segment and export `output/segments_classified.gpkg`.

Load that GeoPackage into QGIS (drag & drop) and color by the
`dump_proba` column (Graduated, threshold ~0.5) to inspect results.

Everything the pipeline does is logged with timestamps to the console,
including warnings for anything that looks off (mismatched mine IDs,
no-data pixels found, a cross-validation fold with only one class,
etc.). Check the log output for warnings after a run.

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
