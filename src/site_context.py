"""
Umgebung der Standorte: Betriebsgelände oder offene Wüste?
==========================================================

Ordnet jedem Standort anhand der Umrisse von Minen/Steinbrüchen (z.B. data/mine_boundaries.gpkg
und data/northChileQuarrys.geojson) eine Umgebung zu:

    inside_op    der Standort liegt (mit seinem Mittelpunkt) im Umriss einer Mine/eines Steinbruchs
    near_op      höchstens --near-m Meter davon entfernt (Rand/Anlagen außerhalb des Umrisses)
    open_desert  weiter weg

Die Spalte `context` (und `op_dist_m`) steuert in QGIS (ai_classifier.py) den Umkreis, den die KI
mitbeurteilt: im Betrieb ein großer (Anlagen, Halden zwischen Gebäuden und Straßen), in der offenen
Wüste ein kleiner. Das braucht weder Bilder noch die API.

Nutzung (vorhandene Liste ergänzen; Urteile, Notizen und die Zeichenschicht bleiben erhalten):
    python src/site_context.py --sites candidate_sites.gpkg --out candidate_sites_ctx.gpkg \\
        --operations data/mine_boundaries.gpkg data/northChileQuarrys.geojson
Beim Bauen neuer Listen: candidate_sites.py ... --operations a.gpkg,b.geojson
"""

import argparse
import logging
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

logger = logging.getLogger(__name__)

CONTEXTS = ("inside_op", "near_op", "open_desert")
POINT_BUFFER_M = 150.0      # Betriebe, die nur als Punkt (OSM-Knoten) vorliegen


def load_operations(paths, crs):
    """Umrisse aus mehreren Dateien (gpkg/geojson) als ein GeoDataFrame im Ziel-CRS; Punkte werden zu
    kleinen Kreisen, Linien/leere Geometrien verworfen."""
    parts = []
    for path in paths:
        g = gpd.read_file(path.strip())
        if g.crs is None:
            g = g.set_crs(4326)
        g = g.to_crs(crs)[["geometry"]]
        pts = g.geometry.geom_type.isin(["Point", "MultiPoint"])
        g.loc[pts, "geometry"] = g.loc[pts, "geometry"].buffer(POINT_BUFFER_M)
        parts.append(g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"]) & ~g.geometry.is_empty])
    ops = pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame(geometry=[], crs=crs)
    return gpd.GeoDataFrame(ops, geometry="geometry", crs=crs)


def add_context(sites, operations, near_m=250.0):
    """Spalten `context` und `op_dist_m` (Abstand des Standorts zum nächsten Umriss, 0 = darin)."""
    sites = sites.copy()
    if sites.empty:
        sites["context"], sites["op_dist_m"] = pd.Series(dtype="object"), pd.Series(dtype="float")
        return sites
    if operations is None or operations.empty:
        sites["context"], sites["op_dist_m"] = "open_desert", float("nan")
        return sites
    ops = operations.to_crs(sites.crs)
    pts = gpd.GeoDataFrame(geometry=sites.geometry.representative_point(), crs=sites.crs)
    inside = gpd.sjoin(pts, ops[["geometry"]], predicate="within", how="left")
    inside = inside[~inside.index.duplicated()]["index_right"].notna()
    near = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=sites.geometry, crs=sites.crs), ops[["geometry"]],
                             how="left", distance_col="_d")
    dist = near.groupby(level=0)["_d"].min().reindex(sites.index)
    dist = dist.where(~inside, 0.0)
    sites["op_dist_m"] = dist.round(1)
    sites["context"] = ["inside_op" if i else ("near_op" if d <= near_m else "open_desert")
                        for i, d in zip(inside, dist)]
    return sites


def main(argv=None):
    p = argparse.ArgumentParser(description="Umgebung (Betrieb/Wüste) zu einer Standortliste ergänzen.")
    p.add_argument("--sites", required=True, help="Standortliste (.gpkg, Schicht candidate_sites).")
    p.add_argument("--out", required=True, help="Ausgabe (neue Datei; die Eingabe bleibt unverändert).")
    p.add_argument("--operations", nargs="+", required=True, help="Umrisse von Minen/Steinbrüchen.")
    p.add_argument("--near-m", type=float, default=250.0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import pyogrio
    layers = [name for name, _ in pyogrio.list_layers(args.sites)]
    sites = gpd.read_file(args.sites, layer="candidate_sites")
    out = add_context(sites, load_operations(args.operations, sites.crs), args.near_m)
    out.to_file(args.out, layer="candidate_sites", driver="GPKG")
    for name in layers:                      # z.B. die Zeichenschicht drawn_polygons unverändert mitnehmen
        if name != "candidate_sites":
            other = gpd.read_file(args.sites, layer=name)
            other.to_file(args.out, layer=name, driver="GPKG", mode="a",
                          geometry_type="Polygon" if len(other) == 0 else None)
    logging.info("%d Standorte: %s -> %s", len(out), out["context"].value_counts().to_dict(), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
