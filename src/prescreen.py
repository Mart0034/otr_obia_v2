"""
Kacheln vor dem Herunterladen nach Landbedeckung aussortieren
=============================================================

Für eine ganze Region (Raster aus Kacheln) bringt es am meisten, Fläche gar nicht erst herunterzuladen, auf der
keine Halde liegen kann: Wasser, Schnee, Wald, Ackerland, Siedlungen. Dazu liest dieses Modul pro Kachel ein winziges
Stück der kostenlosen ESA-WorldCover-Karte (10 m, Planetary Computer; nur eine grobe Vorschau, ein paar Kilobyte)
und berechnet die Anteile der Klassen:

    10 Baum  20 Gebüsch  30 Gras  40 Acker  50 bebaut  60 kahl/spärlich  70 Schnee/Eis  80 Wasser
    90 Feuchtgebiet  95 Mangroven  100 Moos/Flechten

Eine Kachel wird behalten, wenn nichts Ausschließendes überwiegt (Standard: bebaut <= 30 %, Wasser <= 50 %,
Schnee <= 50 %, Baum+Acker+Feuchtgebiet+Mangroven <= 50 %). Die Regeln stehen in RULES und lassen sich ändern.
"""

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)

CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow", 80: "water",
           90: "wetland", 95: "mangrove", 100: "moss"}
RULES = {"max_built": 0.30, "max_water": 0.50, "max_snow": 0.50, "max_green": 0.50}
COLLECTION = "esa-worldcover"


def class_fractions(arr):
    """Anteile der WorldCover-Klassen in einem Array (0 = keine Daten wird ignoriert)."""
    arr = np.asarray(arr)
    valid = arr > 0
    n = int(valid.sum())
    if n == 0:
        return {}
    return {name: float((arr == code).sum()) / n for code, name in CLASSES.items() if (arr == code).any()}


def keep_tile(fractions, rules=None):
    """(behalten?, Grund). Ohne Daten (leeres dict) wird die Kachel sicherheitshalber behalten."""
    rules = {**RULES, **(rules or {})}
    if not fractions:
        return True, "keine Landbedeckungsdaten"
    f = lambda *names: sum(fractions.get(n, 0.0) for n in names)
    checks = (("bebaut", f("built"), rules["max_built"]), ("Wasser", f("water"), rules["max_water"]),
              ("Schnee/Eis", f("snow"), rules["max_snow"]),
              ("Baum/Acker/Feuchtgebiet", f("tree", "crop", "wetland", "mangrove"), rules["max_green"]))
    for name, value, limit in checks:
        if value > limit:
            return False, f"{name} {value:.0%} > {limit:.0%}"
    return True, "ok"


def _read_fractions(catalog, bounds_wgs84, size_px):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    west, south, east, north = bounds_wgs84
    items = list(catalog.search(collections=[COLLECTION], bbox=bounds_wgs84).items())
    if not items:
        return {}
    newest = max(i.properties.get("start_datetime") or "" for i in items)
    merged = np.zeros((size_px, size_px), dtype=np.uint8)
    for it in items:
        if (it.properties.get("start_datetime") or "") != newest:
            continue
        with rasterio.open(it.assets["map"].href) as ds:
            win = from_bounds(west, south, east, north, transform=ds.transform)
            arr = ds.read(1, window=win, out_shape=(size_px, size_px), boundless=True, fill_value=0,
                          resampling=Resampling.mode)
        merged = np.where(merged == 0, arr, merged)
    return class_fractions(merged)


def screen_tiles(tiles, catalog=None, size_px=24, n_threads=8, rules=None, work_crs="EPSG:32719"):
    """tiles: dicts mit center_x, center_y, radius_m (in work_crs). Gibt pro Kachel (behalten, Grund, Anteile) zurück."""
    from pyproj import Transformer
    if catalog is None:
        from fetch_sentinel1 import open_catalog
        catalog = open_catalog()
    to_wgs = Transformer.from_crs(work_crs, "EPSG:4326", always_xy=True)

    def one(t):
        r = t["radius_m"]
        x0, y0 = to_wgs.transform(t["center_x"] - r, t["center_y"] - r)
        x1, y1 = to_wgs.transform(t["center_x"] + r, t["center_y"] + r)
        try:
            frac = _read_fractions(catalog, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), size_px)
        except Exception as e:                    # lieber behalten als fälschlich verwerfen
            return True, f"Landbedeckung nicht lesbar ({type(e).__name__})", {}
        keep, why = keep_tile(frac, rules)
        return keep, why, frac

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        out = list(pool.map(one, tiles))
    kept = sum(1 for k, _, _ in out if k)
    logger.info("Landbedeckungs-Vorauswahl: %d von %d Kacheln behalten (%.0f s).", kept, len(out), time.time() - t0)
    return out


def grid_tiles(west, south, east, north, tile_m, prefix="g", work_crs="EPSG:32719"):
    """Regelmäßiges Raster aus quadratischen Kacheln über ein Gebiet (Längen/Breiten in Grad). Gibt eine Liste von
    dicts im Format des Kachelplans zurück (site_id, center_x, center_y, radius_m, status='ok')."""
    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", work_crs, always_xy=True)
    xs = [to_utm.transform(lon, lat) for lon, lat in ((west, south), (east, south), (west, north), (east, north))]
    x0, x1 = min(p[0] for p in xs), max(p[0] for p in xs)
    y0, y1 = min(p[1] for p in xs), max(p[1] for p in xs)
    radius = math.ceil(tile_m / 2 / 10.0) * 10.0
    side = 2 * radius
    nx, ny = max(1, math.ceil((x1 - x0) / side)), max(1, math.ceil((y1 - y0) / side))
    rows = []
    for j in range(ny):
        for i in range(nx):
            rows.append({"site_id": f"{prefix}_{len(rows):05d}", "center_x": x0 + (i + 0.5) * side,
                         "center_y": y0 + (j + 0.5) * side, "radius_m": radius, "area_m2": side * side,
                         "status": "ok", "buffer_m": 0.0, "members": ""})
    return rows
