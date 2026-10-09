"""
Kandidaten-Standorte aus den Segment-Vorhersagen
================================================

Das Modell bewertet einzelne Segmente. Für die Praxis (zu Fuß prüfen, in
hochauflösenden Bildern ansehen, Größe/Tonnage schätzen) sind aber ZUSAMMEN-
HÄNGENDE Funde interessant: ein Cluster aus 40 hoch bewerteten Segmenten ist
etwas anderes als ein einzelnes. Dieses Skript

  1. wählt die am höchsten bewerteten Segmente je Mine, bis --area-pct % der
     Minenfläche markiert sind (wie im Segmentgrößen-Vergleich) - oder, mit
     --threshold, alle Segmente mit dump_proba >= Schwelle,
  2. verbindet Segmente, die höchstens --join-dist-m auseinanderliegen, zu
     einem Standort,
  3. berechnet je Standort Fläche, mittlere/maximale Bewertung, einen Rang
     (Fläche x mittlere Bewertung) und einen Google-Maps-Link,
  4. schreibt alles als GeoPackage (für QGIS) und als CSV.

Die Datei enthält außerdem eine leere Schicht `drawn_polygons`, in die man in
QGIS die echten Halden-Umrisse einzeichnen kann (siehe unten).

Die Spalten `review` und `note` bleiben leer - in QGIS kann man dort eintragen,
ob ein Fund eine echte Halde ist (dump), sauberes Gelände (clean) oder unklar
(unsure). Das ist die Grundlage, um echte Negativbeispiele zu sammeln.

Nutzung:
    python src/candidate_sites.py --segments out_px15/segments_classified.gpkg \\
        --labels data/dump_labels.gpkg --out out_px15/candidate_sites.gpkg
"""

import argparse
import logging
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Größenklassen nach zusammenhängender Fläche (m²). Zum Vergleich: die Kreis-
# Labels für mittlere/große/sehr große Halden haben ~5.000 / ~15.000 / ~70.000 m².
SIZE_CLASSES = ((2500, "small"), (10000, "medium"), (40000, "large"))
INFO_COLS = ("is_building", "is_poi", "is_road_grid")


def _size_class(area_m2):
    for limit, name in SIZE_CLASSES:
        if area_m2 < limit:
            return name
    return "very_large"


def select_seeds(seg, threshold=None, area_pct=2.0):
    """Segmente, die als Fund gelten: dump_proba >= threshold, oder (ohne
    threshold) je Mine die höchstbewerteten bis area_pct % der gültigen Fläche."""
    seg = seg[seg["dump_proba"].notna()].copy()
    seg["_area"] = seg.geometry.area
    if threshold is not None:
        return seg[seg["dump_proba"] >= threshold]
    keep = []
    for _, grp in seg.groupby("mine_id"):
        grp = grp.sort_values("dump_proba", ascending=False)
        keep.append(grp[grp["_area"].cumsum() - grp["_area"] < grp["_area"].sum() * area_pct / 100.0])
    return pd.concat(keep) if keep else seg.iloc[0:0]


def _components(grp, join_dist_m, crs):
    """Zusammenhängende Gruppen von Segmenten (Abstand <= join_dist_m)."""
    merged = grp.geometry.buffer(join_dist_m / 2.0).union_all()
    comps = gpd.GeoDataFrame(geometry=list(getattr(merged, "geoms", [merged])), crs=crs)
    comps["_comp"] = np.arange(len(comps))
    joined = gpd.sjoin(grp, comps, predicate="intersects", how="inner")
    joined = joined[~joined.index.duplicated(keep="first")]
    return [g.drop(columns=["index_right", "_comp"], errors="ignore") for _, g in joined.groupby("_comp")]


def _split_big(g, join_dist_m, max_area_m2, crs):
    """Zerlegt einen zu großen Standort in seine hochbewerteten Kerne: das
    niedrigste Viertel der Segmente (nach dump_proba) wird so lange abgeschnitten,
    bis die verbleibenden zusammenhängenden Teile höchstens max_area_m2 groß sind.
    Abgeschnitten werden nur die schwach bewerteten Ränder."""
    if not max_area_m2 or g["_area"].sum() <= max_area_m2 or len(g) < 2:
        return [g]
    keep = g[g["dump_proba"] > g["dump_proba"].quantile(0.25)]
    if keep.empty:                      # alle gleich bewertet: nicht weiter teilbar
        return [g]
    out = []
    for comp in _components(keep, join_dist_m, crs):
        out += _split_big(comp, join_dist_m, max_area_m2, crs)
    return out


def build_candidate_sites(seg, threshold=None, area_pct=2.0, join_dist_m=20.0, min_area_m2=0.0,
                          labels=None, known_dist_m=50.0, top=None, max_site_m2=100000.0,
                          exclude_known=False, max_urban_frac=None, rank_by="score", exclude_areas=None,
                          sample_every=None, sample_skip=0):
    """seg: Segmente mit mine_id, dump_proba, geometry (projiziertes CRS in Metern).
    Gibt ein GeoDataFrame mit einem Standort pro Zeile zurück, nach Rang sortiert.
    max_site_m2: größere Standorte werden in ihre Kerne zerlegt (0 = aus).
    exclude_known: Standorte nahe bekannter Halden (labels) weglassen.
    rank_by: Sortierung - "score" (Fläche x mittlere Bewertung), "max_proba" (höchste
    Segment-Bewertung im Standort) oder "mean_proba". In der ersten Prüfung von 300
    Standorten waren die Treffer bei max_proba >= 0.95 am häufigsten.
    exclude_areas: GeoDataFrame mit Flächen (z.B. bereits geprüfte Standorte): Standorte,
    die eine davon berühren, werden weggelassen.
    max_urban_frac: Standorte weglassen, bei denen mehr als dieser Flächenanteil
    Gebäude / Straßenraster / Sehenswürdigkeit laut OSM ist (nur mit den
    OSM-Spalten im Lauf, siehe --osm-*-dir der Pipeline)."""
    if seg.crs is None or not seg.crs.is_projected:
        raise ValueError("Die Segmente brauchen ein projiziertes CRS (Meter), z.B. UTM.")
    seeds = select_seeds(seg, threshold, area_pct)
    parts = []
    for mine, grp in seeds.groupby("mine_id"):
        for comp in _components(grp, join_dist_m, seg.crs):
            for g in _split_big(comp, join_dist_m, max_site_m2, seg.crs):
                area = float(g["_area"].sum())
                row = {
                    "mine_id": mine,
                    "n_segments": int(len(g)),
                    "area_m2": area,
                    "mean_proba": float((g["dump_proba"] * g["_area"]).sum() / area),
                    "max_proba": float(g["dump_proba"].max()),
                    "geometry": g.geometry.union_all(),
                }
                for col in INFO_COLS:
                    if col in g.columns:
                        row[f"frac_{col}"] = float((g[col].astype(float) * g["_area"]).sum() / area)
                parts.append(row)
    cols = ["mine_id", "n_segments", "area_m2", "mean_proba", "max_proba"]
    if not parts:
        return gpd.GeoDataFrame(columns=["site_id", "rank", *cols, "geometry"],
                                geometry="geometry", crs=seg.crs)
    sites = gpd.GeoDataFrame(parts, geometry="geometry", crs=seg.crs)
    sites = sites[sites["area_m2"] >= min_area_m2]
    sites["score"] = sites["area_m2"] * sites["mean_proba"]
    frac_cols = [c for c in sites.columns if c.startswith("frac_")]
    if frac_cols:
        sites["urban_frac"] = sites[frac_cols].max(axis=1).round(3)
        if max_urban_frac is not None:
            sites = sites[sites["urban_frac"] <= max_urban_frac]
    elif max_urban_frac is not None:
        logger.warning("--max-urban-frac ignoriert: der Lauf enthält keine OSM-Spalten "
                       "(is_building/is_road_grid/is_poi).")
    if exclude_areas is not None and len(exclude_areas) and len(sites):
        hit = gpd.sjoin(sites[["geometry"]], exclude_areas.to_crs(sites.crs)[["geometry"]],
                        predicate="intersects", how="inner")
        sites = sites[~sites.index.isin(hit.index)].reset_index(drop=True)
    if rank_by not in ("score", "max_proba", "mean_proba"):
        raise ValueError("rank_by muss score, max_proba oder mean_proba sein")
    sites = sites.sort_values([rank_by, "score"], ascending=False).reset_index(drop=True)

    if labels is not None and len(labels):
        lab = labels.to_crs(seg.crs)
        near = gpd.sjoin_nearest(sites[["geometry"]], lab[["geometry"]], how="left",
                                 distance_col="_d")
        d = near.groupby(level=0)["_d"].min()
        sites["near_known_dump"] = (d.reindex(sites.index) <= known_dist_m).values
        if exclude_known:
            sites = sites[~sites["near_known_dump"]].reset_index(drop=True)
    if top:
        sites = sites.head(top)
    sites.insert(0, "rank", np.arange(1, len(sites) + 1))
    sites.insert(1, "site_id", [f"S{r:04d}" for r in sites["rank"]])
    if sample_every and sample_every > 1:
        # Stichprobe: Ränge <= sample_skip (schon geprüft) weglassen, danach jeder sample_every-te.
        # rank/site_id behalten ihren Wert aus der vollen Liste -> Trefferquote je Rang bleibt auswertbar.
        keep = (sites["rank"] > sample_skip) & ((sites["rank"] - sample_skip) % sample_every == 0)
        sites = sites[keep].reset_index(drop=True)
    sites["size_class"] = [_size_class(a) for a in sites["area_m2"]]

    pts = sites.geometry.representative_point()
    ll = gpd.GeoSeries(pts, crs=seg.crs).to_crs(4326)
    sites["lat"], sites["lon"] = ll.y.round(6).values, ll.x.round(6).values
    sites["maps_url"] = [
        f"https://www.google.com/maps/search/?api=1&query={la},{lo}"
        for la, lo in zip(sites["lat"], sites["lon"])
    ]
    sites["review"] = ""
    sites["note"] = ""
    return sites


def load_top_segments(path, area_pct=2.0, threshold=None, fraction_margin=1.5):
    """Liest nur die Segmente, die als Fund in Frage kommen - ohne die ganze (oft mehrere GB
    große) Datei in den Speicher zu laden: erst nur die Tabelle OHNE Geometrie (schnell), dann je
    Mine die bestbewerteten Segmente nach Anzahl (area_pct * fraction_margin % der Segmente)
    bzw. alle ab threshold, und nur deren Geometrien über die Objekt-IDs.
    Die Auswahl nach Anzahl statt nach Fläche weicht nur dort ab, wo die Segmente einer Mine
    sehr unterschiedlich groß sind; die Flächenauswahl von select_seeds läuft danach noch einmal."""
    import pyogrio

    attrs = pyogrio.read_dataframe(path, read_geometry=False, fid_as_index=True)
    attrs = attrs[attrs["dump_proba"].notna()]
    if threshold is not None:
        keep = attrs[attrs["dump_proba"] >= threshold]
    else:
        parts = []
        for _, grp in attrs.groupby("mine_id"):
            n = max(1, int(np.ceil(len(grp) * area_pct * fraction_margin / 100.0)))
            parts.append(grp.nlargest(n, "dump_proba"))
        keep = pd.concat(parts) if parts else attrs.iloc[0:0]
    seg = pyogrio.read_dataframe(path, fids=keep.index.values)
    logger.info("%d von %d Segmenten geladen (Vorauswahl je Mine).", len(seg), len(attrs))
    return seg


def write_sites_gpkg(sites, path):
    """Schreibt die Standorte und eine zusätzliche, leere Polygon-Schicht
    `drawn_polygons` in dieselbe GeoPackage-Datei. In QGIS werden dort die
    tatsächlichen Halden-Umrisse (kind = dump) bzw. eindeutig saubere Flächen
    (kind = clean) von Hand eingezeichnet; reviewed_to_labels.py liest sie."""
    if os.path.exists(path):
        os.remove(path)
    sites.to_file(path, layer="candidate_sites", driver="GPKG")
    empty = gpd.GeoDataFrame(
        {"kind": pd.Series(dtype="object"), "mine_id": pd.Series(dtype="object"),
         "note": pd.Series(dtype="object")},
        geometry=gpd.GeoSeries([], crs=sites.crs))
    empty.to_file(path, layer="drawn_polygons", driver="GPKG", mode="a", geometry_type="Polygon")


def sites_from_segments(seg, out, threshold=None, area_pct=2.0, join_dist_m=20.0, min_area_m2=0.0,
                        labels_path=None, known_dist_m=50.0, top=None, max_site_m2=100000.0,
                        exclude_known=False, max_urban_frac=None, rank_by="score", exclude_areas=None,
                        sample_every=None, sample_skip=0):
    """Standortliste aus bewerteten Segmenten (seg: mine_id, dump_proba, geometry) bauen und als
    GeoPackage + CSV schreiben. Wird vom Kommandozeilen-Skript und - mit den Segmenten direkt aus dem
    Speicher - von der Pipeline selbst benutzt (--candidate-lists), ohne die Segmentdatei neu zu lesen.
    exclude_areas: Dateien (mit Komma getrennt) mit Flächen, die nicht mehr angezeigt werden sollen."""
    labels = gpd.read_file(labels_path) if labels_path else None
    excl = None
    if exclude_areas:
        excl = gpd.GeoDataFrame(pd.concat(
            [gpd.read_file(f.strip())[["geometry"]].to_crs(seg.crs) for f in exclude_areas.split(",") if f.strip()],
            ignore_index=True), crs=seg.crs)
    sites = build_candidate_sites(seg, threshold, area_pct, join_dist_m, min_area_m2, labels, known_dist_m, top,
                                  max_site_m2, exclude_known, max_urban_frac, rank_by, excl,
                                  sample_every, sample_skip)
    write_sites_gpkg(sites, out)
    csv_path = out.rsplit(".", 1)[0] + ".csv"
    pd.DataFrame(sites.drop(columns="geometry")).to_csv(csv_path, index=False)
    logging.info("%d Standorte -> %s (+ %s)", len(sites), out, csv_path)
    if len(sites):
        logging.info("Größenklassen: %s", sites["size_class"].value_counts().to_dict())
    return sites


def main(argv=None):
    p = argparse.ArgumentParser(description="Kandidaten-Standorte aus Segment-Vorhersagen bauen.")
    p.add_argument("--segments", required=True, help="segments_classified.gpkg eines Laufs.")
    p.add_argument("--out", required=True, help="Ausgabe-GeoPackage (zusätzlich eine .csv daneben).")
    p.add_argument("--area-pct", type=float, default=2.0,
                   help="Je Mine die höchstbewerteten Segmente bis zu diesem %% der Fläche markieren "
                        "(Standard 2). Wird ignoriert, wenn --threshold gesetzt ist.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Stattdessen: alle Segmente mit dump_proba >= Schwelle markieren.")
    p.add_argument("--join-dist-m", type=float, default=20.0,
                   help="Segmente bis zu diesem Abstand gehören zum selben Standort.")
    p.add_argument("--min-area-m2", type=float, default=0.0, help="Kleinere Standorte weglassen.")
    p.add_argument("--top", type=int, default=None, help="Nur die besten N Standorte.")
    p.add_argument("--sample-every", type=int, default=None,
                   help="Nur jeden N-ten Standort der Rangliste ausgeben (Stichprobe, um die Trefferquote "
                        "über viele Ränge mit wenig Prüfaufwand zu schätzen). Mit --top = Tiefe der Liste.")
    p.add_argument("--sample-skip", type=int, default=0,
                   help="Zusammen mit --sample-every: die ersten N Ränge auslassen (schon geprüft).")
    p.add_argument("--labels", default=None,
                   help="Bekannte Halden (GeoPackage); markiert Standorte nahe bekannter Halden.")
    p.add_argument("--known-dist-m", type=float, default=50.0)
    p.add_argument("--max-site-m2", type=float, default=100000.0,
                   help="Größere Standorte in ihre hochbewerteten Kerne zerlegen (0 = nicht zerlegen). "
                        "Standard 100000 m2; echte Halden sind meist 5.000 - 70.000 m2.")
    p.add_argument("--max-urban-frac", type=float, default=None,
                   help="Standorte weglassen, die zu mehr als diesem Anteil (0-1) aus OSM-Gebäuden, "
                        "Straßenraster oder Sehenswürdigkeiten bestehen (z.B. 0.2). Braucht einen "
                        "Lauf mit --osm-buildings-dir/--osm-roads-dir/--osm-poi-dir.")
    p.add_argument("--rank-by", choices=["score", "max_proba", "mean_proba"], default="score",
                   help="Reihenfolge der Standorte: score = Fläche x mittlere Bewertung (Standard), "
                        "max_proba = höchste Segment-Bewertung (bisher treffsicherer).")
    p.add_argument("--exclude-areas", default=None,
                   help="Dateien (mit Komma getrennt) mit Flächen, die nicht mehr angezeigt werden sollen, "
                        "z.B. bereits geprüfte Standorte (negatives_reviewed_clean.gpkg, ignore_reviewed.geojson).")
    p.add_argument("--preselect", action="store_true",
                   help="Für sehr große Dateien (mehrere GB): erst nur die Bewertungen lesen und dann nur "
                        "die Geometrien der bestbewerteten Segmente laden (spart viel Speicher).")
    p.add_argument("--exclude-known", action="store_true",
                   help="Standorte nahe bekannter Halden (--labels) weglassen - zum Suchen NEUER Halden.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Mit --preselect ist die Auswahl (je Mine die obersten area_pct % der Segmente) schon beim
    # Laden erfolgt; danach werden alle geladenen Segmente verwendet.
    seg = load_top_segments(args.segments, args.area_pct, args.threshold, fraction_margin=1.0) \
        if args.preselect else gpd.read_file(args.segments)
    sites_from_segments(
        seg, args.out, threshold=args.threshold, area_pct=100.0 if args.preselect else args.area_pct,
        join_dist_m=args.join_dist_m, min_area_m2=args.min_area_m2, labels_path=args.labels,
        known_dist_m=args.known_dist_m, top=args.top, max_site_m2=args.max_site_m2,
        exclude_known=args.exclude_known, max_urban_frac=args.max_urban_frac, rank_by=args.rank_by,
        exclude_areas=args.exclude_areas, sample_every=args.sample_every, sample_skip=args.sample_skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
