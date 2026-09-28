# Pterodactyl egg: OTR-OBIA processing server

`egg-otr-obia-pipeline.json` is a Pterodactyl "Egg" (server template) for
running this pipeline on a dedicated Pterodactyl-managed VM/box, separate
from the interactive session Claude runs in.

## What it sets up

- Runtime image: `python:3.11-slim-bookworm` (or the 3.12 variant, selectable
  in the panel when creating the server).
- Install step: installs `git` + build tools + GDAL/GEOS/PROJ system
  libraries, clones the repo, creates a venv at `/home/container/venv`, and
  `pip install`s everything in `requirements.txt` (rasterio, geopandas,
  scikit-learn, imbalanced-learn, pvlib, pystac-client, planetary-computer,
  ...).
- Startup: **not** a long-running service. It activates the venv and drops
  into a persistent `bash` on the console, so pipeline runs are triggered by
  sending shell commands to that console (via the panel or the Client API),
  e.g.:

  ```
  cd src && python3 otr_obia_pipeline.py --imagery-dir ../data/imagery \
    --labels-path ../data/dump_labels.gpkg \
    --mine-boundary-path ../data/mine_boundaries.gpkg \
    --s1-dir ../data/sentinel1 --dem-dir ../data/dem --s2t-dir ../data/s2_temporal \
    --require-neighbor-below 0.6 --require-compact-cluster-below 0.8 \
    --output-dir ../output_final
  ```

  Console output (stdout/stderr) streams back the normal way, so progress
  and results can be read the same way as any other Pterodactyl server log.

## Importing

In the panel: **Admin → Nests → your nest → Import Egg**, upload
`egg-otr-obia-pipeline.json` (or point the importer at the raw GitHub URL of
this file, if the panel supports import-by-URL). No secrets/API keys are
baked into it — Sentinel-1/2, Copernicus DEM (Microsoft Planetary Computer)
and ESA WorldCover are all fetched without authentication, so the only
per-server configuration is the four egg variables (repo URL, branch,
auto-update, requirements file).

## Notes

- No egress data volumes are pre-provisioned; `data/` is populated by
  running the `src/fetch_*.py` scripts after first boot (or by syncing a
  pre-fetched `data/` directory onto the server).
- `AUTO_UPDATE=1` makes every server (re)start `git pull` + reinstall
  requirements — handy while iterating, but leave it off if you want a
  server pinned to a known-good commit.
