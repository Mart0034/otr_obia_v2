"""
Daten für die öffentliche Demo-Seite (docs/) exportieren - mit absichtlich ungenauen Orten
===========================================================================================

Die Seite liegt öffentlich auf GitHub Pages. Genaue Koordinaten von Halden auf Minengelände sollen dort NICHT
stehen. Dieses Skript schreibt deshalb nur:

  * Orte von Halden/Kandidaten auf ein 0,01°-Raster (ca. 1,1 km) gerundet und innerhalb einer Zelle zusammengefasst,
    nie die Umrisse oder die Standort-IDs,
  * die Minen-Umrisse der 138 bekannten Minen (öffentliche Daten), vereinfacht,
  * Kennzahlen (stats.json) aus den geprüften Listen.

Nutzung (Pfade der geprüften Dateien anpassen):
    python scripts/export_site_data.py --out docs/data \\
        --reviewed r1:mines=a.gpkg r2:quarries=b.gpkg ... \\
        --candidates output_corr/candidate_sites_v8_newchile_only.gpkg \\
        --known data/dump_labels.gpkg --mines data/mine_boundaries.gpkg
"""

import argparse
import json
import math
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

GRID_DEG = 0.01
RECENT_ROUNDS = ("r07", "r08", "r09")      # Runden mit dem verbesserten KI-Prompt (v6, Neu-Chile, v7)
REGIONS = (("north", "Norte (Arica - Tarapacá)", -22.6, -17.0),
           ("atacama", "Atacama (Antofagasta)", -25.7, -22.6),
           ("south", "Sur (Atacama - Coquimbo)", -33.0, -25.7))


def snap(lon, lat, grid=GRID_DEG):
    """Mitte der Rasterzelle, in der der Punkt liegt (gerundet auf 5 Nachkommastellen)."""
    return (round((math.floor(lon / grid) + 0.5) * grid, 5), round((math.floor(lat / grid) + 0.5) * grid, 5))


def region_of(lat):
    for key, _, south, north in REGIONS:
        if south <= lat < north:
            return key
    return "other"


def size_class(area_m2):
    return "s" if area_m2 < 8000 else "m" if area_m2 < 16000 else "l"


def round5(p):
    return None if p is None or pd.isna(p) else int(round(float(p) * 20) * 5)


def how_found(row):
    """Wie wurde der Fund gemacht? ai: das KI-Urteil lag bei 50 % oder mehr (System + KI, Mensch bestätigt);
    human: die KI war unsicher/dagegen - oder es gab die KI-Prüfung noch nicht (frühe Runden) - und der Mensch hat
    es erkannt."""
    p = row.get("ai_dump_p")
    return "ai" if p is not None and not pd.isna(p) and float(p) >= 0.5 else "human"


def cells(df, props):
    """Punkte (lon/lat) auf Zellen runden und zusammenfassen: pro Zelle und Gruppe eine Zeile mit Anzahl."""
    rows = {}
    for _, r in df.iterrows():
        key = (*snap(r["lon"], r["lat"]), *(r[k] for k in props))
        rows.setdefault(key, []).append(r)
    feats = []
    for key, rs in rows.items():
        lon, lat = key[0], key[1]
        p = {k: v for k, v in zip(props, key[2:])}
        p["n"] = len(rs)
        p["region"] = region_of(lat)
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": p})
    return {"type": "FeatureCollection", "features": feats}


def load_reviewed(spec):
    """spec: Einträge "Runde:Art=Datei" in zeitlicher Reihenfolge (z.B. "r1:mines=a.gpkg"). Gibt die Standorte aller
    Runden zurück (ein Standort, der in mehreren Listen steht, zählt einmal: die spätere Beurteilung gilt)."""
    frames = []
    for item in spec:
        head, path = item.split("=", 1)
        label, _, tag = head.partition(":")
        g = gpd.read_file(path, layer="candidate_sites")
        c = g.to_crs(4326).geometry.centroid
        g["lon"], g["lat"] = c.x, c.y
        g["round"], g["tag"], g["listed"] = label, tag or "", len(g)
        for col in ("ai_dump_p", "ai_decision"):
            if col not in g.columns:
                g[col] = None
        frames.append(pd.DataFrame(g.drop(columns="geometry")))
    d = pd.concat(frames, ignore_index=True)
    d["review"] = d["review"].fillna("").astype(str).str.lower()
    d["ai_decision"] = d["ai_decision"].fillna("").astype(str)
    d = d[d["review"].isin(["dump", "partial", "clean", "unsure"])].copy()
    d["_key"] = list(zip(d["lon"].round(4), d["lat"].round(4), d["area_m2"].round(-1)))
    d = d.drop_duplicates("_key", keep="last").drop(columns="_key")
    d["auto"] = d["ai_decision"] == "auto_clean"          # vom Plugin automatisch verworfen, nicht von einem Menschen gesehen
    return d


def round_table(frames_spec, reviewed):
    """Pro Runde: Listenlänge, davon von einem Menschen geprüft, automatisch verworfen, bestätigt."""
    order = [item.split("=", 1)[0] for item in frames_spec]
    rows = []
    for head in order:
        label, _, tag = head.partition(":")
        g = reviewed[reviewed["round"] == label]
        human = g[~g["auto"]]
        rows.append({"round": label, "tag": tag, "listed": int(g["listed"].iloc[0]) if len(g) else 0,
                     "checked": int(len(human)), "auto": int(g["auto"].sum()),
                     "confirmed": int(human["review"].isin(["dump", "partial"]).sum())})
    return rows


def stats_from(reviewed, rounds):
    d = reviewed[reviewed["review"].isin(["dump", "partial", "clean"])].copy()
    d["pos"] = d["review"].isin(["dump", "partial"])
    human = reviewed[~reviewed["auto"]]
    n_human, n_pos = int(len(human)), int(human["review"].isin(["dump", "partial"]).sum())
    ai = d[d["ai_dump_p"].notna()]
    hi, lo = ai[ai["ai_dump_p"] >= 0.7], ai[ai["ai_dump_p"] < 0.3]
    recent = ai[ai["round"].isin(RECENT_ROUNDS)]
    rhi = recent[recent["ai_dump_p"] >= 0.7]
    spot = reviewed[reviewed["ai_decision"].isin(["spot_ok", "spot_miss"])]
    spot_miss = int(((spot["ai_decision"] == "spot_miss") & spot["review"].isin(["dump", "partial"])).sum())
    return {
        "reviewed": n_human, "confirmed": n_pos, "hit_rate": round(n_pos / max(n_human, 1), 4), "rounds": rounds,
        "ai_high": {"n": int(len(hi)), "confirmed": int(hi["pos"].sum()), "share": round(float(hi["pos"].mean()), 3)},
        "ai_low": {"n": int(len(lo)), "confirmed": int(lo["pos"].sum()), "share": round(float(lo["pos"].mean()), 4)},
        "ai_high_recent": {"n": int(len(rhi)), "confirmed": int(rhi["pos"].sum()),
                           "share": round(float(rhi["pos"].mean()), 3) if len(rhi) else None},
        "auto_clean": int(reviewed["auto"].sum()),
        "spot_checks": int(len(spot)), "spot_misses": spot_miss,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/data")
    p.add_argument("--reviewed", nargs="+", required=True, help="Runde=Datei (geprüfte Kandidatenlisten)")
    p.add_argument("--candidates", help="Noch ungeprüfte Kandidaten (z.B. v8 nur Neu-Chile).")
    p.add_argument("--open-round", help="Art der noch offenen Runde (zeigt sie im Diagramm als offen), z.B. chile_all.")
    p.add_argument("--known", required=True, help="Von Hand markierte Halden-Polygone (dump_labels.gpkg).")
    p.add_argument("--mines", required=True, help="Umrisse der bekannten Minen.")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    reviewed = load_reviewed(args.reviewed)
    pos = reviewed[reviewed["review"].isin(["dump", "partial"])].copy()
    pos["how"] = pos.apply(how_found, axis=1)
    pos["size"] = pos["area_m2"].map(size_class)
    pos["kind"] = pos["review"]
    found = cells(pos, ["kind", "how", "size"])
    # KI-Wert je Zelle als Mittel (auf 5 % gerundet), ohne einzelne Standorte erkennbar zu machen
    json.dump(found, open(os.path.join(args.out, "found.json"), "w"), separators=(",", ":"))

    known = gpd.read_file(args.known)
    kc = known.to_crs(4326).geometry.centroid
    kdf = pd.DataFrame({"lon": kc.x, "lat": kc.y, "size": known.to_crs(32719).area.map(size_class)})
    known_cells = cells(kdf, ["size"])
    json.dump(known_cells, open(os.path.join(args.out, "known.json"), "w"), separators=(",", ":"))

    cand_cells = {"type": "FeatureCollection", "features": []}
    if args.candidates:
        c = gpd.read_file(args.candidates)
        cc = c.to_crs(4326).geometry.centroid
        cdf = pd.DataFrame({"lon": cc.x, "lat": cc.y, "size": c["area_m2"].map(size_class),
                            "score": (c["mean_proba"] * 20).round().astype(int) * 5})
        cand_cells = cells(cdf, ["size", "score"])
    json.dump(cand_cells, open(os.path.join(args.out, "candidates.json"), "w"), separators=(",", ":"))

    mines = gpd.read_file(args.mines).to_crs(32719)
    mines["geometry"] = mines.geometry.simplify(40)
    mines = mines[~mines.geometry.is_empty].to_crs(4326)[["geometry"]]
    mines.to_file(os.path.join(args.out, "mines.json"), driver="GeoJSON", COORDINATE_PRECISION=4)

    rounds = round_table(args.reviewed, reviewed)
    if args.open_round:
        n_open = int(sum(f["properties"]["n"] for f in cand_cells["features"]))
        rounds.append({"round": "r%02d" % (len(rounds) + 1), "tag": args.open_round, "listed": n_open, "checked": 0,
                       "auto": 0, "confirmed": 0, "open": True})
    st = stats_from(reviewed, rounds)
    st["known_polygons"] = int(len(known))
    st["mines"] = int(len(mines))
    st["found_cells"] = len(found["features"])
    st["candidates"] = int(sum(f["properties"]["n"] for f in cand_cells["features"]))
    st["found_by_how"] = {k: int((pos["how"] == k).sum()) for k in ("ai", "human")}
    st["regions"] = {}
    for key, name, south, north in REGIONS:
        in_region = lambda d: int(((d["lat"] >= south) & (d["lat"] < north)).sum())
        st["regions"][key] = {
            "name": name, "south": south, "north": north,
            "known": in_region(kdf), "found": in_region(pos),
            "found_ai": in_region(pos[pos["how"] == "ai"]), "found_human": in_region(pos[pos["how"] == "human"]),
            "reviewed": in_region(reviewed[~reviewed["auto"]]),
            "candidates": int(sum(f["properties"]["n"] for f in cand_cells["features"]
                                  if south <= f["geometry"]["coordinates"][1] < north)),
        }
    json.dump(st, open(os.path.join(args.out, "stats.json"), "w"), indent=1, ensure_ascii=False)
    # Alle Daten zusätzlich als eine JS-Datei: so läuft die Seite auch direkt aus dem Ordner (file://), nicht nur über http(s)
    bundle = {"stats": st, "found": found, "known": known_cells, "cand": cand_cells,
              "mines": json.load(open(os.path.join(args.out, "mines.json")))}
    with open(os.path.join(args.out, "data.js"), "w") as f:
        f.write("window.OTR_DATA = " + json.dumps(bundle, separators=(",", ":"), ensure_ascii=False) + ";\n")
    # Versionsnummer an die Skript-Adressen hängen, damit Browser (und GitHub Pages) nie alte Daten/Skripte weiterverwenden
    import re, time
    index = os.path.join(os.path.dirname(os.path.abspath(args.out)), "index.html")
    if os.path.exists(index):
        html = open(index, encoding="utf-8").read()
        html = re.sub(r'((?:assets/(?:style\.css|i18n\.js|app\.js)|data/data\.js))(\?v=\w+)?"', r'\1?v=' + time.strftime("%Y%m%d%H%M") + '"', html)
        open(index, "w", encoding="utf-8").write(html)
    print(json.dumps({k: st[k] for k in ("reviewed", "confirmed", "hit_rate", "ai_high", "ai_low", "auto_clean",
                                          "spot_checks", "spot_misses", "known_polygons", "mines", "candidates",
                                          "found_by_how")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
