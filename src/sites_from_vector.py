"""
Kacheln für viele neue Stellen aus einer Vektordatei bauen
============================================================

Nimmt eine Vektordatei (GeoPackage/GeoJSON/Shapefile, z.B. der Export von
Steinbrüchen aus overpass-turbo oder QGIS) und macht daraus pro Objekt eine
Kachel im Format von data/imagery/*.tif, damit die Pipeline diese Stellen mit
--extra-imagery-dir prüfen kann (siehe README, "Screening a brand-new site").

Pro Objekt: Geometrie um --buffer-m Meter erweitern, das Quadrat um die
erweiterte Ausdehnung ist die Kachel (Kantenlänge = längere Seite der
Ausdehnung, mindestens 2*--min-radius-m). Optional Breitengrad-Filter
(--south/--north) und Mindestfläche (--min-area-m2).

Ohne --fetch wird nur der Plan geschrieben (sites.csv) - zum Prüfen, wie viele
Kacheln entstehen würden und welche zu groß sind. Mit --fetch werden zusätzlich
Sentinel-2-, Sentinel-1- und DEM-Kacheln geladen (--with-osm: auch OSM).

Nutzung:
    python src/sites_from_vector.py --input quarries.geojson --buffer-m 500 \\
        --south -26 --north -21 --out-dir data/sites_quarries
    python src/sites_from_vector.py ... --fetch
"""

import argparse
import logging
import math
import os
import re
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

logger = logging.getLogger(__name__)

WORK_CRS = "EPSG:32719"
PIXEL_M = 10.0


def _site_id(row, position, prefix):
    for col in ("@id", "id", "osm_id", "full_id"):
        if col in row.index and pd.notna(row[col]):
            return f"{prefix}_" + re.sub(r"[^A-Za-z0-9]+", "_", str(row[col])).strip("_")
    return f"{prefix}_{position:05d}"


def _merge_overlapping(plan, prefix, max_tile_m):
    """Fasst sich überlappende Kacheln (Status ok) zu einer gemeinsamen Kachel
    zusammen - die Bounding-Box des Clusters. Cluster, deren Kachel größer als
    max_tile_m würde, bleiben unverändert (einzelne Kacheln)."""
    ok = plan[plan["status"] == "ok"]
    if len(ok) < 2:
        return plan
    squares = gpd.GeoDataFrame(
        {"i": ok.index},
        geometry=[box(r.center_x - r.radius_m, r.center_y - r.radius_m,
                      r.center_x + r.radius_m, r.center_y + r.radius_m) for r in ok.itertuples()],
    )
    parent = {i: i for i in ok.index}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # "overlaps" statt "intersects": bloßes Berühren am Rand zählt nicht
    hits = squares.sjoin(squares, predicate="intersects")
    for a, b in zip(hits["i_left"], hits["i_right"]):
        if a != b and squares.geometry.loc[squares["i"] == a].iloc[0].intersection(
                squares.geometry.loc[squares["i"] == b].iloc[0]).area > 0:
            parent[find(a)] = find(b)

    groups = {}
    for i in ok.index:
        groups.setdefault(find(i), []).append(i)
    keep, merged_rows, k = [], [], 0
    for members in groups.values():
        if len(members) == 1:
            continue
        sub = squares[squares["i"].isin(members)]
        minx, miny, maxx, maxy = sub.total_bounds
        radius = math.ceil(max(maxx - minx, maxy - miny) / 2 / PIXEL_M) * PIXEL_M
        if 2 * radius > max_tile_m:
            continue
        k += 1
        rows = plan.loc[members]
        merged_rows.append({
            "site_id": f"{prefix}_cluster_{k:03d}", "center_x": (minx + maxx) / 2,
            "center_y": (miny + maxy) / 2, "radius_m": radius,
            "area_m2": rows["area_m2"].sum(), "status": "ok",
            "members": ",".join(rows["site_id"]),
        })
        keep.extend(members)
    if not merged_rows:
        return plan
    return pd.concat([plan.drop(index=keep), pd.DataFrame(merged_rows)], ignore_index=True)


def plan_sites(gdf, buffer_m=0.0, south=None, north=None, min_area_m2=0.0,
               max_tile_m=10000.0, min_radius_m=500.0, prefix="q", work_crs=WORK_CRS,
               merge_overlaps=False):
    """Gibt ein DataFrame mit einer Zeile pro Objekt zurück: site_id, center_x,
    center_y (in work_crs), radius_m (halbe Kantenlänge, auf 10 m gerundet),
    area_m2 und status ("ok" oder der Grund, warum es übersprungen wird)."""
    if gdf.crs is None:
        raise ValueError("Die Eingabedatei hat kein Koordinatensystem (CRS).")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    rows = []
    proj = gdf.to_crs(work_crs)
    lat = proj.geometry.centroid.to_crs("EPSG:4326").y
    for pos, (idx, row) in enumerate(gdf.iterrows()):
        geom = proj.geometry.loc[idx]
        status = "ok"
        if south is not None and lat.loc[idx] < south:
            status = "südlich des Breitengrad-Filters"
        elif north is not None and lat.loc[idx] > north:
            status = "nördlich des Breitengrad-Filters"
        elif geom.area < min_area_m2:
            status = "unter Mindestfläche"
        minx, miny, maxx, maxy = geom.buffer(buffer_m).bounds
        side = max(maxx - minx, maxy - miny, 2 * min_radius_m)
        radius = math.ceil(side / 2 / PIXEL_M) * PIXEL_M
        if status == "ok" and 2 * radius > max_tile_m:
            status = f"Kachel zu groß ({2 * radius / 1000:.1f} km > {max_tile_m / 1000:.1f} km)"
        rows.append({
            "site_id": _site_id(row, pos, prefix),
            "center_x": (minx + maxx) / 2, "center_y": (miny + maxy) / 2,
            "radius_m": radius, "area_m2": geom.area, "status": status,
            "members": "",
        })
    plan = pd.DataFrame(rows)
    if not plan.empty and plan["site_id"].duplicated().any():
        dup = plan["site_id"].duplicated(keep=False)
        plan.loc[dup, "site_id"] = [f"{s}_{i}" for i, s in enumerate(plan.loc[dup, "site_id"])]
    if merge_overlaps:
        plan = _merge_overlapping(plan, prefix, max_tile_m)
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(description="Kacheln aus einer Vektordatei (z.B. Steinbrüche) bauen.")
    p.add_argument("--input", required=True, help="Vektordatei (GeoJSON, GPKG, SHP, ...).")
    p.add_argument("--out-dir", default="data/sites", metavar="DIR",
                   help="Zielordner; darin imagery/, sentinel1/, dem/ und sites.csv.")
    p.add_argument("--buffer-m", type=float, default=0.0, help="Erweiterung jedes Objekts in Metern.")
    p.add_argument("--south", type=float, help="Südgrenze (Breitengrad, z.B. -26).")
    p.add_argument("--north", type=float, help="Nordgrenze (Breitengrad, z.B. -21).")
    p.add_argument("--min-area-m2", type=float, default=0.0, help="Kleinere Objekte überspringen.")
    p.add_argument("--max-tile-m", type=float, default=10000.0, help="Größere Kacheln überspringen.")
    p.add_argument("--min-radius-m", type=float, default=500.0, help="Mindest-Halbkantenlänge.")
    p.add_argument("--merge-overlaps", action="store_true",
                   help="Sich überlappende Kacheln zu einer zusammenfassen (spart doppeltes "
                        "Laden/Bewerten); Cluster über --max-tile-m bleiben einzeln.")
    p.add_argument("--prefix", default="q", help="Präfix der Kachelnamen (site_id).")
    p.add_argument("--fetch", action="store_true", help="Kacheln auch wirklich laden.")
    p.add_argument("--with-osm", action="store_true",
                   help="Mit --fetch: auch OSM-Gebäude/Straßen/POI laden (braucht Zugriff auf overpass-api.de).")
    p.add_argument("--n-workers", type=int, default=4,
                   help="Anzahl gleichzeitig geladener Kacheln (das Laden ist netzwerkbegrenzt).")
    p.add_argument("--limit", type=int, help="Nur die ersten N passenden Objekte laden (zum Ausprobieren).")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    gdf = gpd.read_file(args.input)
    plan = plan_sites(
        gdf, buffer_m=args.buffer_m, south=args.south, north=args.north,
        min_area_m2=args.min_area_m2, max_tile_m=args.max_tile_m,
        min_radius_m=args.min_radius_m, prefix=args.prefix,
        merge_overlaps=args.merge_overlaps,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    plan.to_csv(os.path.join(args.out_dir, "sites.csv"), index=False)
    ok = plan[plan["status"] == "ok"]
    logger.info("%d Objekte gelesen, %d Kacheln geplant.", len(plan), len(ok))
    for status, n in plan[plan["status"] != "ok"]["status"].str.replace(r"\(.*\)", "", regex=True).value_counts().items():
        logger.info("  übersprungen (%s): %d", status.strip(), n)
    if len(ok):
        logger.info("Kachelgröße: Median %.1f km, max %.1f km; geschätzt %.1f Mio Pixel gesamt.",
                    2 * ok["radius_m"].median() / 1000, 2 * ok["radius_m"].max() / 1000,
                    ((2 * ok["radius_m"] / PIXEL_M) ** 2).sum() / 1e6)
    if not args.fetch:
        logger.info("Nur Plan geschrieben (%s/sites.csv). Mit --fetch werden die Kacheln geladen.", args.out_dir)
        return 0

    if args.limit:
        ok = ok.head(args.limit)
    from fetch_s2_base_imagery import fetch_for_site
    from fetch_sentinel1 import open_catalog

    imagery, s1, dem = (os.path.join(args.out_dir, d) for d in ("imagery", "sentinel1", "dem"))
    os.makedirs(imagery, exist_ok=True)
    open_catalog()  # richtet Logging + GDAL-Pfade einmal ein
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    local = threading.local()

    def build(s):
        out = os.path.join(imagery, f"{s['site_id']}.tif")
        if os.path.exists(out):
            return s["site_id"], "vorhanden"
        if not hasattr(local, "catalog"):
            local.catalog = open_catalog()
        tmp = out + ".part"
        try:
            msg = fetch_for_site(s["center_x"], s["center_y"], s["radius_m"], tmp,
                                 local.catalog, crs=WORK_CRS)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        os.replace(tmp, out)  # nur fertige Kacheln tragen den endgültigen Namen
        return s["site_id"], msg

    failed = []
    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(build, s): s["site_id"] for _, s in ok.iterrows()}
        for fut in as_completed(futures):
            try:
                logger.info("%s: %s", *fut.result())
            except Exception as e:
                logger.error("%s: fehlgeschlagen (%s)", futures[fut], e)
                failed.append(futures[fut])
    import fetch_dem
    import fetch_sentinel1
    workers = str(args.n_workers)
    fetch_sentinel1.main(["--imagery-dir", imagery, "--out-dir", s1, "--n-workers", workers])
    fetch_dem.main(["--imagery-dir", imagery, "--out-dir", dem, "--n-workers", workers])
    if args.with_osm:
        import fetch_osm_buildings
        import fetch_osm_poi
        import fetch_osm_roads
        for mod, name in ((fetch_osm_buildings, "osm_buildings"), (fetch_osm_roads, "osm_roads"),
                          (fetch_osm_poi, "osm_poi")):
            mod.main(["--imagery-dir", imagery, "--out-dir", os.path.join(args.out_dir, name)])
    if failed:
        logger.warning("%d Kacheln fehlgeschlagen: %s", len(failed), failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
