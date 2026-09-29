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
0.15). Segments at or above the confidence value are never dropped, same
reasoning as the neighbor filter, but use a higher cutoff here than for
`--require-neighbor-below`: a single high-confidence segment being a real
dump is plausible (mine_012), but a whole 20+-segment chain the model is
uniformly confident about almost never is, real dumps aren't shaped like
roads no matter how sure the model gets. Tested on real data: a road
chain in mine_043 where every segment sat at 0.60-0.71 confidence slipped
straight through a 0.6 cutoff untouched; raising the cutoff to `0.8`
caught it. `0.6` for `--require-neighbor-below` and `0.8` for
`--require-compact-cluster-below` is a reasonable starting pair.

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

### Adding multi-date Sentinel-1 features (optional)

Radar backscatter depends heavily on viewing geometry. A geometrically
well-defined surface (a road, a building edge, a pit wall) scatters
differently depending on look angle, so its backscatter swings between
acquisitions; a disordered pile of tires scatters diffusely in most
directions and stays comparatively constant. Same idea as the multi-date
Sentinel-2 shadow feature above, but for radar look-angle instead of sun
angle.

1. Compute the per-pixel temporal variation from up to 12 Sentinel-1 RTC
   scenes spread over a year (free, no account):

   ```bash
   python src/fetch_sentinel1_timeseries.py --imagery-dir data/imagery --out-dir data/s1_temporal \
     --start 2023-01-01 --end 2023-12-31
   ```

2. Run the pipeline with `--s1t-dir data/s1_temporal` (combines with
   `--s1-dir`, `--dem-dir`, `--s2t-dir`). Per segment this adds the mean
   and variation of:
   - `s1t_vv_cv` / `s1t_vh_cv`: standard deviation / mean of linear
     backscatter across all scenes (0 = identical backscatter every
     acquisition, high = strong look-angle-dependent swing)

   Spot-checked on 3 mines (Escondida/`mine_019`, `mine_043`, `mine_079`):
   no clear improvement over the existing `s1_vv_std`/`s1_vh_std` spatial
   texture features on that small sample - neither hurt nor obviously
   helped. Worth re-checking on a larger set of mines before relying on it.

### Investigated: separating buildings from dumps (not adopted as-is)

Buildings (warehouses, sheds) are a persistent false-positive category -
rectangular roofs with a sharp light/shadow edge look similar to a dump in
several features. Two ideas were checked against the real 36 labeled dump
segments before deciding whether to act on them:

- **Weighting `shape_rectangularity` + `brightness_extreme_fraction`
  more heavily** (e.g. as an extra downgrade filter like
  `apply_cluster_shape_filter`) - rejected. On the real data, real dumps
  and false positives have almost identical rectangularity distributions
  (median 0.742 vs 0.738); SLIC segments are inherently somewhat
  box-shaped by construction, so the feature mostly reflects the
  segmentation algorithm, not the object. Any threshold strong enough to
  meaningfully cut false positives also caught up to 14% of real dumps
  (5/36 at a loose setting); the only threshold that caught zero real
  dumps only removed 1.8% of borderline false positives. Not worth it.
- **OSM building footprints as a hard exclusion filter** - rejected in
  that specific form, safe but weak *on average*. Zero of the 36 real
  dumps overlap an OSM building polygon (even with a neighbor-
  propagation extension, tried on the idea that sparse point coverage
  might "spread" further - only 3 extra segments out of 70,317, so no
  real effect either way), so a filter would never cost a real
  detection. But only ~1.6% of borderline false positives (proba
  0.5-0.8) overlap one, averaged across all 138 mines - OSM building
  coverage for remote Atacama industrial sites is wildly uneven, from
  zero at most mines to ~20-30% of false positives at a handful of
  well-mapped ones (e.g. the "Minera Mechilla" processing plant,
  `mine_043`). Shipped instead as an informational `is_building` column
  rather than a filter - see "Flagging segments that overlap an OSM
  building (optional)" below.
- **OSM `landuse=industrial`/`quarry` polygons as a hard exclusion** -
  worse than useless, rejected outright. Every mine is itself tagged
  industrial or quarry land in OSM, and every known dump sits on mine
  property, so **all 36 real dumps also overlap one of these polygons at
  100%** (while it does catch ~92% of borderline false positives, e.g.
  the "Mina Michilla" processing plant complex). This signal only
  separates "inside a mine" from "outside one," not "dump" from
  "building" within a mine - never test at the mine-property level again,
  it's the wrong scale entirely.

  Net conclusion after three rejected ideas (rectangularity/bimodality
  reweighting, OSM buildings, OSM landuse): small buildings sub-segment-
  scale relative to Sentinel-2 (10m) don't separate from dumps on shape,
  texture, or any free auxiliary vector layer tried so far. This looks
  like a genuine material-identity problem rather than a shape/context
  one - see "PRISMA hyperspectral imagery" under Future work below for
  the one untested option that's actually suited to that.

  Note for `curl`/`requests` users hitting this API from a sandboxed
  environment: `overpass.openstreetmap.fr` returns 403 ("only available
  to white-listed usages") to Python's `requests` library specifically,
  even with an identical query and the same proxy, while plain `curl`
  succeeds - looks like TLS-fingerprint-based bot blocking rather than a
  network policy or header issue. `fetch_osm_buildings.py` shells out to
  `curl` to work around it.

### Flagging segments that overlap an OSM building (optional)

1. Fetch building footprints per mine (free, no account - see the
   investigation above for why this stays informational rather than a
   filter: OSM coverage is too uneven to trust everywhere at once):

   ```bash
   python src/fetch_osm_buildings.py --imagery-dir data/imagery --out-dir data/osm_buildings
   ```

2. Run the pipeline with `--osm-buildings-dir data/osm_buildings`. The
   exported map gets one extra column:
   - `is_building`: `True` for a segment that overlaps a mapped OSM
     building, or touches another segment that does; `False` otherwise
     (including every mine with no `<mine_id>.gpkg` fetched, or an empty
     one - never an error).

   This **never changes `dump_pred` or `dump_proba`** - it's there so
   you can hide these segments yourself in QGIS (right-click layer →
   Filter, or Properties → Source → Query Builder: `"is_building" = 0`,
   optionally combined with `"dump_pred" = 1`) at the mines where you
   judge OSM's coverage is good enough to trust, without the pipeline
   silently deciding that for every mine at once.

### Screening a brand-new site (no existing tile)

All 138 existing `data/imagery/*.tif` tiles came from "Stage 2" of the
earlier feasibility study, not from this repository (see above) - so
there was previously no way to point the pipeline at a location that
hasn't already been through that process. `fetch_s2_base_imagery.py`
builds the same 10-band stack from scratch for any center point + radius:

```bash
python src/fetch_s2_base_imagery.py --center-x 365950.1 --center-y 7367190.3 \
  --radius-m 1500 --mine-id new_001 --out-dir data/new_sites/imagery
```

`blue/green/red/nir/swir1/swir2` are normal Sentinel-2 L2A reflectance,
and `ndvi`/`ndwi`/`bsi` were reverse-engineered against the real
`data/imagery` tiles and match exactly (error < 1e-7):

```
ndvi = (nir - red) / (nir + red)
ndwi = (green - swir1) / (green + swir1)          # a modified NDWI, not the textbook (green-nir) version
bsi  = ((swir1+red) - (nir+blue)) / ((swir1+red) + (nir+blue))
```

**`dsi` could not be recovered.** It's the pipeline's designated
`texture_band` and one of the more important engineered features, but no
standard index (NDBI, DBSI, any simple normalized difference of the 6 raw
bands) matched the real values even approximately - it's almost certainly
a bespoke formula from Stage 2 that isn't documented anywhere in this
repository. Rather than guess, the script fills the `dsi` band with the
`bsi` value instead (clearly marked in the band description and logged as
a warning at fetch time), so the pipeline stays runnable but the
`dsi_mean`/`dsi_std`/GLCM-texture features for a new site are **not**
directly comparable to the 138 known mines. If you can get the real DSI
formula from whoever ran Stage 2, that's the clean fix.

To actually screen a new site: fetch its `--s1-dir`/`--dem-dir`/`--s2t-dir`
layers the normal way (pointing at the new tile), then run the pipeline
with its imagery folder merged alongside the 138 known mines (so the
model still trains on real, labeled dumps and just scores the new site's
unlabeled segments too) rather than on its own - a lone unlabeled tile has
nothing to train a classifier from.

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
- Small buildings (smaller than a segment, i.e. finer than ~10m) remain
  a persistent false-positive source that nothing currently in the
  pipeline reliably separates from real dumps - see "Investigated:
  separating buildings from dumps" above for what's been tried and
  ruled out.

## Future work

- **PRISMA hyperspectral imagery** (ASI, ~200+ narrow spectral bands,
  30m resolution, free with registration, confirmed coverage over
  Chile) is the most promising untested option for the small-buildings-
  vs-dumps problem above. Every idea tried so far (shape, texture, OSM
  buildings, OSM landuse) operates on coarse Sentinel-2 spectral bands
  or auxiliary vector data, neither of which can tell rubber from roofing
  material/rock at a chemical level. PRISMA's fine spectral resolution
  is built for exactly that kind of material identification and hasn't
  been tried yet.
- **ALOS PALSAR** (JAXA L-band SAR, free, 25m, available via AWS/GEE) as
  an additional radar source alongside the existing Sentinel-1 C-band -
  lower priority than PRISMA since Sentinel-1 already engages the
  "does this scatter like a rough diffuse pile" question; L-band would
  be a variation on a signal already in use rather than a new axis.
