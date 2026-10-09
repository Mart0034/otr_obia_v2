"""
OBIA-Pipeline für OTR-Reifenhalden-Erkennung
=============================================

Alternative zum Pixel-Segmentierungsansatz (U-Net/DeepLabV3+) aus der
Feasibility Study. Statt jedes Pixel einzeln zu klassifizieren, wird
das Bild pro Mine in homogene Segmente (Superpixel) zerlegt und JEDES
SEGMENT als Ganzes klassifiziert (Object-Based Image Analysis, OBIA).

Vorteile gegenüber dem CNN-Ansatz bei nur 12 positiven Minen:
  - aus 12 Minen entstehen hunderte/tausende Segment-Beispiele
    (statt 12 Rasterbilder) -> deutlich mehr Trainingsdaten für den
    Klassifikator
  - Random Forest funktioniert robust auch mit wenigen hundert
    Beispielen und wenigen Merkmalen, im Gegensatz zu einem CNN
  - kein GPU-Training, kein Chip-Tiling, kein Class-Balancing nötig

Workflow:
  1. Sentinel-2-Stack pro Mine laden (10 Bänder, wie in Stage 2 des
     bestehenden Projekts bereits vorhanden)
  2. SLIC-Superpixel-Segmentierung pro Mine
  3. Segmente in Polygone umwandeln
  4. pro Segment: spektrale Mittel-/Std-Werte je Band + GLCM-Textur
     + Formmerkmale berechnen
  5. Segmente anhand der in QGIS digitalisierten Dump-Polygone labeln
     (Overlap-Kriterium)
  6. Random Forest trainieren, Validierung per GroupKFold NACH MINE
     (gleiches Prinzip wie der räumliche Split im bestehenden Report,
     verhindert Data Leakage)
  7. alle Segmente klassifizieren und als GeoPackage exportieren ->
     kann direkt in QGIS geladen und visuell geprüft werden

Abhängigkeiten (NICHT die QGIS-Bundle-Python-Umgebung, sondern eigenes
venv/conda-env empfohlen, da schwere ML-Libs im QGIS-Python oft
Probleme machen):

    pip install rasterio geopandas shapely scikit-image scikit-learn pandas numpy

Nutzung in QGIS: das erzeugte GeoPackage (output/segments_classified.gpkg)
einfach per Drag&Drop in QGIS laden und nach der Spalte "dump_proba"
einfärben (Graduated, z. B. 0.5 als Schwelle).

Erwartete Eingangsdaten:
  - imagery_dir: ein GeoTIFF pro Mine (Dateiname enthält mine_id),
    10 Bänder in der Reihenfolge aus CONFIG["band_names"]
  - labels_path: GeoPackage/Shapefile mit den in QGIS digitalisierten
    Dump-Polygonen, Spalte mine_id_field verknüpft Label mit Mine
"""

import os
import glob
import re
import logging
import multiprocessing
from concurrent.futures.process import BrokenProcessPool
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.ndimage import find_objects
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.warp import transform as warp_transform
from scipy.ndimage import uniform_filter
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union
from skimage.segmentation import slic
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import pvlib

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# KONFIGURATION - hier an die eigenen Pfade/Daten anpassen
# ------------------------------------------------------------------
CONFIG = {
    "imagery_dir": "data/imagery",              # ein GeoTIFF pro Mine
    "labels_path": "data/dump_labels.gpkg",      # digitalisierte Dump-Polygone
    "mine_id_field": "mine_id",                  # Spaltenname in labels_path
    "n_segments_per_mine": 800,                  # SLIC-Zielanzahl Superpixel,
                                                   # NUR verwendet wenn
                                                   # target_segment_px=None
    # target_segment_px/max_segments_per_mine: bei den echten Antofagasta-
    # Daten reichen die Minenbilder von ~72.000 bis ~5.860.000 Pixeln (Faktor
    # 80), ein fester n_segments_per_mine erzeugt dadurch bei großen Minen
    # riesige Segmente (bis zu 7300 Pixel) gegenüber oft nur 1-10 Pixel
    # großen Dump-Polygonen -> praktisch nie genug Überlappung für ein
    # positives Label. target_segment_px berechnet n_segments stattdessen
    # pro Mine aus der Bildgröße (gültige Pixel / target_segment_px),
    # begrenzt durch max_segments_per_mine, damit große Minen die Laufzeit
    # nicht explodieren lassen. Werte unten sind ein erster, an den echten
    # Dump-Größen kalibrierter Versuch (siehe Datenanalyse), keine final
    # getunten Werte.
    "target_segment_px": 30,
    "max_segments_per_mine": 4000,
    "compactness": 8,                             # SLIC: Form- vs. Farbtreue
    "min_overlap_ratio": 0.10,                    # Segment = positiv, wenn
                                                   # >= 10% seiner Fläche im
                                                   # gelabelten Dump liegt.
                                                   # Abgesenkt von 0.30: die
                                                   # meisten echten Dump-
                                                   # Polygone sind kleiner als
                                                   # ein einzelnes Segment,
                                                   # 30% Überlappung war real
                                                   # kaum je erreichbar.
    "band_names": [
        "blue", "green", "red", "nir", "swir1", "swir2",
        "ndvi", "ndwi", "dsi", "bsi",
    ],
    "texture_band": "dsi",                        # Band für GLCM-Textur
    "n_estimators": 400,
    "random_state": 42,
    "n_jobs": None,                               # Minen parallel verarbeiten:
                                                   # None = alle CPU-Kerne nutzen,
                                                   # 1 = sequentiell (alter Modus)
    "feature_workers": None,                      # Prozesse für die Merkmals-
                                                   # berechnung EINER großen Mine
                                                   # (None = alle CPU-Kerne)
    "use_cache": False,                           # zwischengespeicherten
                                                   # Segment-Datensatz aus
                                                   # output_dir wiederverwenden
                                                   # statt neu zu berechnen
    # Optional: Pfad zu einer Datei mit den tatsächlichen Minen-Grenzen
    # (z.B. mines_antofagasta.shp), um Segmente außerhalb der Mine
    # auszuschließen (siehe filter_to_mine_boundary()). None = kein Filter,
    # exakt das bisherige Verhalten.
    "mine_boundary_path": None,
    "mine_boundary_id_field": "mine_id",
    "mine_boundary_buffer_m": 0.0,
    # Schwellwert für dump_proba -> dump_pred (0/1), sowohl in der
    # Kreuzvalidierung als auch im finalen Export. War bisher fest auf 0.5
    # codiert; bei extrem unausgeglichenen Daten (1:4850) ist das ein
    # schlechter Kompromiss (viele False Positives). Getestet an den echten
    # Daten: 0.8 statt 0.5 senkt False Positives um ca. Faktor 4-5, bei nur
    # leicht geringerer Erkennungsquote (6 statt 9 von 9 Minen). Konfigurierbar,
    # damit man das ohne Codeänderung anpassen kann (siehe --use-cache, um
    # dafür nicht jedes Mal neu zu segmentieren).
    "classification_threshold": 0.5,
    # Optional: erfordert für Segmente mit dump_proba UNTER diesem Wert
    # mindestens ein räumlich angrenzendes, ebenfalls positiv vorhergesagtes
    # Segment in derselben Mine (siehe apply_neighbor_filter()). Zielt auf
    # isolierte Einzelsegmente mittlerer Konfidenz entlang von Fahrstraßen/
    # Gruben-Rändern, die den Großteil der Falsch-Positiven ausmachen.
    # Segmente mit dump_proba >= diesem Wert bleiben unangetastet, auch ohne
    # Nachbarn - manche echten Dumps bestehen aus nur einem Segment. None =
    # kein Filter, exakt das bisherige Verhalten.
    "require_neighbor_below": None,
    # Optional: erfordert für Segmente mit dump_proba UNTER diesem Wert, dass
    # ihr zusammenhängendes Cluster berührender positiver Segmente in
    # derselben Mine eine Kompaktheit >= min_cluster_compactness hat (siehe
    # apply_cluster_shape_filter()). Zielt auf lange, dünne Ketten entlang
    # von Straßen/Klippenkanten, die den Nachbarschafts-Filter oben unbeschadet
    # passieren (die Segmente unterstützen sich ja gegenseitig), aber nicht
    # haufenförmig wie ein echter Dump sind. Segmente mit dump_proba >= diesem
    # Wert bleiben unangetastet. None = kein Filter.
    "require_compact_cluster_below": None,
    "min_cluster_compactness": 0.15,
    # Optional: Ordner mit Sentinel-1-Radardaten pro Mine (erzeugt von
    # src/fetch_sentinel1.py, gleiche Dateinamen und gleiches Pixelraster wie
    # die Sentinel-2-Bilder). None = keine Radar-Merkmale.
    "s1_dir": None,
    # Optional: Ordner mit dem Höhenmodell pro Mine (erzeugt von
    # src/fetch_dem.py, gleiches Pixelraster wie Sentinel-2). None = keine
    # Gelände-Merkmale.
    "dem_dir": None,
    # Fenstergröße (Pixel) für die Lage relativ zur Umgebung (dem_tpi):
    # 31 px = ~300m bei 10m-Pixeln.
    "dem_tpi_window_px": 31,
    # Optional: Ordner mit Zeitreihen-Merkmalen aus mehreren Sentinel-2-
    # Aufnahmen übers Jahr (erzeugt von src/fetch_s2_timeseries.py). None =
    # keine Zeitreihen-Merkmale.
    "s2t_dir": None,
    # Optional: Ordner mit Rückstreu-Zeitreihen-Merkmalen aus mehreren
    # Sentinel-1-Aufnahmen (erzeugt von src/fetch_sentinel1_timeseries.py).
    # None = keine Radar-Zeitreihen-Merkmale.
    "s1t_dir": None,
    # Optional: Ordner mit Mehrjahres-Merkmalen aus Sentinel-2 (erzeugt von
    # src/fetch_s2_multiyear.py): wächst an der Stelle etwas Dunkles? None = aus.
    "s2my_dir": None,
    "s2my_bands": None,   # Teilmenge der sechs Mehrjahres-Merkmale (Komma-Liste), None = alle
    # Optional: Ordner mit OSM-Gebäudeumrissen pro Mine (erzeugt von
    # src/fetch_osm_buildings.py). Fügt der exportierten Karte eine rein
    # informative is_building-Spalte hinzu (kein Filter, ändert dump_pred
    # nicht - siehe compute_is_building()). None = keine is_building-Spalte.
    "osm_buildings_dir": None,
    "osm_poi_dir": None,
    "poi_buffer_m": 20,
    # Optional: Ordner mit OSM-Straßen pro Mine (erzeugt von
    # src/fetch_osm_roads.py). Fügt der exportierten Karte die rein
    # informativen Spalten road_density_150m/is_road_grid hinzu (kein
    # Filter, ändert dump_pred nicht - siehe compute_road_density()).
    # None = keine dieser Spalten.
    "osm_roads_dir": None,
    "extra_imagery_dir": None,
    "extra_labels_path": None,
    "labeled_imagery_dir": None,
    "reviewed_imagery_dir": None,                 # Kacheln, von denen nur geprüfte Stellen trainiert werden
    "hard_negatives_path": None,                  # bestätigte Fehlalarme (Polygone, mine_id):
                                                   # ihre Segmente zählen im Training mehrfach
    "hard_negative_factor": 10,                   # ... und zwar so oft
    "add_labels_path": None,                      # zusätzliche Halden-Polygone (mine_id =
                                                   # Mine/Kachel), z.B. aus reviewed_to_labels.py:
                                                   # werden auf den Datensatz angewendet
    "ignore_points_path": None,
    "ignore_radius_m": 150.0,
    "road_density_radius_m": 150,
    "road_grid_threshold_m": 500,
    "output_dir": "output",
}

# Bänder in den Sentinel-1-Dateien (dB) plus das beim Laden berechnete
# VH/VV-Verhältnis (in dB eine Differenz).
S1_BAND_NAMES = ["s1_vv", "s1_vh", "s1_vh_vv"]

# Aus dem Höhenmodell abgeleitete Merkmale. Die absolute Höhe ist bewusst
# NICHT dabei: sie reicht je nach Mine von der Küste bis ~4000m und würde vor
# allem verraten, um welche Mine es sich handelt, statt etwas über Halden.
# dem_hillshade_dec/_jun: siehe hillshade() weiter unten.
DEM_BAND_NAMES = ["dem_slope", "dem_tpi", "dem_northness", "dem_hillshade_dec", "dem_hillshade_jun"]

# Repräsentative Sonnenstände für die Hillshade-Merkmale: Süd-Sommer- und
# Süd-Wintersonnenwende, je nahe dem lokalen Mittag in Antofagasta (UTC-3/-4).
# Das genaue Jahr ist unerheblich - es geht nur um die zwei Extreme des
# Sonnenstands übers Jahr (siehe fetch_s2_timeseries.py), nicht um eine
# bestimmte echte Aufnahme.
SOLSTICE_DATETIMES = {
    "dec": "2023-12-21T15:00:00Z",  # Süd-Sommer, hoher Sonnenstand
    "jun": "2023-06-21T15:00:00Z",  # Süd-Winter, tiefer Sonnenstand
}

# Zeitreihen-Merkmale: wie stark schwankt die Helligkeit eines Pixels übers
# Jahr (Variationskoeffizient) und wie dunkel wird es im dunkelsten Moment
# im Verhältnis zu seinem Normalwert (Minimum / Median). Schatten wandern mit
# dem Sonnenstand, Reifen nicht.
S2T_BAND_NAMES = ["s2t_bright_cv", "s2t_bright_min_ratio"]

# Radar-Zeitreihen-Merkmal: wie stark schwankt die Rückstreuung eines Pixels
# über mehrere Aufnahmen (Variationskoeffizient). Geometrisch klare Flächen
# (Straßen, Kanten) sind blickwinkelabhängig und schwanken stark, ein
# ungeordneter Reifenhaufen streut diffus und bleibt vergleichsweise
# konstant. Siehe fetch_sentinel1_timeseries.py.
S1T_BAND_NAMES = ["s1t_vv_cv", "s1t_vh_cv"]

# Mehrjahres-Merkmale (siehe fetch_s2_multiyear.py): wächst an einer Stelle etwas
# Dunkles? Helligkeit früh/spät, Veränderung, Steigung, Anteil dunkler Jahre und
# Zeitpunkt des ersten dunklen Jahres.
S2MY_BAND_NAMES = ["s2my_bright_early", "s2my_bright_late", "s2my_bright_delta", "s2my_bright_slope",
                   "s2my_dark_years_frac", "s2my_dark_onset"]


# ------------------------------------------------------------------
# 1) Rasterdaten laden
# ------------------------------------------------------------------
def load_mine_raster(path, expected_n_bands=None):
    """Lädt einen Sentinel-2-Stack (Bänder, Höhe, Breite) + Georeferenz."""
    with rasterio.open(path) as src:
        arr = src.read()  # shape: (bands, H, W)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    arr = np.moveaxis(arr, 0, -1)  # -> (H, W, bands)

    if expected_n_bands is not None and arr.shape[-1] != expected_n_bands:
        raise ValueError(
            f"{path}: erwartet {expected_n_bands} Bänder (siehe CONFIG['band_names']), "
            f"aber im GeoTIFF sind {arr.shape[-1]} Bänder vorhanden."
        )
    logger.debug("%s geladen: shape=%s, crs=%s, nodata=%s", path, arr.shape, crs, nodata)
    return arr, transform, crs, nodata


def extract_mine_id(filename, pattern=r"(mine[_\-]?\d+|\d+)"):
    """Versucht, die mine_id aus dem Dateinamen zu extrahieren.
    Bei Bedarf an das tatsächliche Namensschema anpassen."""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(pattern, base, flags=re.IGNORECASE)
    if pattern == r"(mine[_\-]?\d+|\d+)":
        # Ein "mine_NNN" irgendwo im Namen hat Vorrang vor beliebigen Ziffern
        # davor (z.B. das "2" in "S2_mine_012_2023").
        m = re.search(r"mine[_\-]?\d+", base, flags=re.IGNORECASE) or m
    mine_id = m.group(0) if m else base
    # Reine Ziffern aus einem längeren Namen (new_002, q_way_1111) sind keine
    # Mine-ID: sie würden zu "002"/"1111" und nicht zur passenden OSM-Datei
    # <Name>.gpkg passen bzw. bei gleichen Ziffern kollidieren.
    if m is not None and mine_id.isdigit() and mine_id != base:
        mine_id = base
    if m is None:
        logger.warning(
            "Konnte keine mine_id aus Dateiname '%s' extrahieren, verwende den "
            "vollständigen Dateinamen ('%s') als mine_id. Prüfe ggf. das "
            "Namensschema/die pattern-Regex.", filename, mine_id,
        )
    return mine_id


# ------------------------------------------------------------------
# 2) Segmentierung
# ------------------------------------------------------------------
# Blockweise Segmentierung ab dieser Segmentzahl (siehe segment_mine()).
SLIC_BLOCK_MIN_SEGMENTS = 6000
SLIC_BLOCK_PX = 256


def segment_mine(arr, n_segments, compactness, nodata=None,
                  target_segment_px=None, max_segments=None):
    """SLIC-Superpixel-Segmentierung auf allen Bändern gleichzeitig.
    Ersetzt die GRASS-i.segment-Variante aus QGIS/GRASS, funktioniert
    aber identisch im Prinzip (Region-Growing/Clustering) und ist
    reiner Python-Code ohne GRASS-Abhängigkeit.

    Ist target_segment_px gesetzt, wird n_segments IGNORIERT und
    stattdessen pro Mine aus der Anzahl gültiger Pixel berechnet
    (valid_pixels / target_segment_px), begrenzt durch max_segments.
    Das ist wichtig, weil ein fester n_segments-Wert bei Minen mit
    stark unterschiedlicher Bildgröße (hier: 80x-Unterschied) völlig
    unterschiedlich große Segmente erzeugt - bei großen Minen viel zu
    große Segmente im Vergleich zu den oft nur wenige Pixel großen
    Dump-Polygonen."""
    # Gültige Pixel bestimmen (kein NoData, keine NaN/Inf-Werte) -> werden
    # von der Segmentierung ausgeschlossen, damit keine "Fantasie-Segmente"
    # aus randlichen/fehlenden Bildbereichen entstehen und die spätere
    # Normierung nicht verzerrt wird.
    valid_mask = np.all(np.isfinite(arr), axis=-1)
    if nodata is not None:
        valid_mask &= ~np.all(arr == nodata, axis=-1)
    if not valid_mask.any():
        raise ValueError("Rasterbild enthält keine gültigen (Nicht-NoData) Pixel.")

    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        logger.info(
            "%d von %d Pixeln als NoData/ungültig erkannt und von der "
            "Segmentierung ausgeschlossen.", n_invalid, valid_mask.size,
        )

    if target_segment_px is not None:
        n_valid = int(valid_mask.sum())
        computed = max(20, round(n_valid / target_segment_px))
        if max_segments is not None:
            computed = min(computed, max_segments)
        logger.debug(
            "target_segment_px=%d -> n_segments=%d (statt fixem Wert %d)",
            target_segment_px, computed, n_segments,
        )
        n_segments = computed

    # NaNs/Inf robust behandeln (nur zur Absicherung; die als NoData
    # markierten Pixel fließen dank 'mask' unten ohnehin nicht ins Ergebnis ein)
    safe_arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # SLIC erwartet vernünftig skalierte Werte -> pro Band normieren
    # (Perzentile nur über gültige Pixel, damit NoData die Skalierung nicht verzerrt)
    norm = np.zeros_like(safe_arr, dtype=np.float32)
    for b in range(safe_arr.shape[-1]):
        band = safe_arr[..., b]
        valid_band = band[valid_mask]
        lo, hi = np.percentile(valid_band, [2, 98])
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        norm[..., b] = np.clip((band - lo) / (hi - lo), 0, 1)

    if n_segments <= SLIC_BLOCK_MIN_SEGMENTS:
        return slic(
            norm,
            n_segments=n_segments,
            compactness=compactness,
            channel_axis=-1,
            start_label=1,
            enforce_connectivity=True,
            mask=valid_mask,
        )

    # Sehr viele Segmente: skimage platziert die Startpunkte bei einer Maske per
    # k-means über ALLE gültigen Pixel - Aufwand ~ Pixel x Segmente. Bei einer
    # 2.5-Mpx-Mine mit 80.000 Segmenten läuft das über Stunden (und wurde auf
    # dem Server hart beendet). Deshalb blockweise segmentieren: jeder Block
    # bekommt proportional zu seinen gültigen Pixeln seine Segmentzahl. Die
    # Segmente enden dann an den Blockgrenzen (harmlos bei Halden von wenigen
    # Pixeln Größe).
    height, width = valid_mask.shape
    n_valid_total = int(valid_mask.sum())
    segments = np.zeros((height, width), dtype=np.int32)
    offset = 0
    for y0 in range(0, height, SLIC_BLOCK_PX):
        for x0 in range(0, width, SLIC_BLOCK_PX):
            block_mask = valid_mask[y0:y0 + SLIC_BLOCK_PX, x0:x0 + SLIC_BLOCK_PX]
            n_block_valid = int(block_mask.sum())
            if n_block_valid == 0:
                continue
            n_block = max(1, round(n_segments * n_block_valid / n_valid_total))
            block = norm[y0:y0 + SLIC_BLOCK_PX, x0:x0 + SLIC_BLOCK_PX]
            if n_block < 2:
                seg = np.where(block_mask, 1, 0)
            else:
                seg = slic(
                    block, n_segments=n_block, compactness=compactness, channel_axis=-1,
                    start_label=1, enforce_connectivity=True, mask=block_mask,
                )
            seg = np.where(seg > 0, seg + offset, 0)
            offset = max(offset, int(seg.max()))
            segments[y0:y0 + SLIC_BLOCK_PX, x0:x0 + SLIC_BLOCK_PX] = seg
    return segments


def segments_to_polygons(segments, transform, crs, mine_id):
    """Wandelt die Segment-Label-Matrix in Polygone um (ein Polygon pro
    Segment-ID)."""
    mask = segments > 0
    records = []
    for geom, seg_id in rio_shapes(segments.astype(np.int32), mask=mask, transform=transform):
        records.append({
            "segment_id": int(seg_id),
            "mine_id": mine_id,
            "geometry": shapely_shape(geom),
        })
    gdf = gpd.GeoDataFrame(records, crs=crs)
    # SLIC kann Segmente in mehrere disjunkte Polygone aufteilen -> dissolven
    gdf = gdf.dissolve(by=["mine_id", "segment_id"], as_index=False)
    logger.debug("%s: %d Segment-Polygone erzeugt (crs=%s)", mine_id, len(gdf), crs)
    return gdf


# ------------------------------------------------------------------
# 3) Merkmale pro Segment
# ------------------------------------------------------------------
def compute_shape_features(seg_gdf):
    """Berechnet geometrische Formmerkmale je Segment: Fläche, Umfang,
    Kompaktheit (Polsby-Popper-Maß: 1.0 = perfekter Kreis, kleiner = länglich
    oder verwinkelt) und Rechteckigkeit (Flächenanteil am eigenen minimalen
    umschließenden Rechteck: nah an 1.0 = füllt sein Rechteck fast komplett
    aus wie ein Gebäude/Dach, deutlich darunter = unregelmäßige, organische
    Form wie ein Reifenhaufen). Gebäude auf dem Minengelände wurden beim
    manuellen Durchsehen der Ergebnisse als eigene Falsch-Positiv-Kategorie
    identifiziert - geometrisch klar von Halden unterscheidbar, aber bisher
    nicht gemessen (shape_compactness erfasst Rundheit, nicht Eckigkeit)."""
    seg_gdf = seg_gdf.copy()
    seg_gdf["shape_area"] = seg_gdf.geometry.area
    seg_gdf["shape_perimeter"] = seg_gdf.geometry.length
    seg_gdf["shape_compactness"] = (
        4 * np.pi * seg_gdf["shape_area"] / (seg_gdf["shape_perimeter"] ** 2 + 1e-9)
    )

    def rectangularity(geom):
        rect_area = geom.minimum_rotated_rectangle.area
        return geom.area / rect_area if rect_area > 0 else 0.0

    seg_gdf["shape_rectangularity"] = seg_gdf.geometry.apply(rectangularity)
    return seg_gdf


def brightness_extreme_fraction(vals):
    """Anteil der Pixel, die im UNTEREN oder OBEREN Drittel der eigenen
    (min-max-normierten) Spanne des Segments liegen - hoch, wenn ein
    Segment scharf zweigeteilt ist (z. B. helles Blechdach direkt neben
    dunklem Schatten/Wasser, wie bei überdachten Lagerhallen oder
    ausgekleideten Absetzbecken), niedrig bei einer breiten, gleichmäßig
    verteilten Textur (wie bei einer Reifenhalde). Anders als tex_contrast
    (GLCM, misst lokale Nachbarschaftsunterschiede im Mittel) erfasst das
    gezielt eine BIMODALE Verteilung, unabhängig davon, wie die beiden
    Bereiche räumlich angeordnet sind."""
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return 0.0
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-6:
        return 0.0
    norm = (vals - lo) / (hi - lo)
    return float(((norm <= 1 / 3) | (norm >= 2 / 3)).mean())


# Ab so vielen Segmenten lohnt es sich, die Merkmalsberechnung einer Mine auf
# mehrere Prozesse zu verteilen (darunter ist der Start der Prozesse teurer).
PARALLEL_MIN_SEGMENTS = 20000

# Wird vor dem Start der Kind-Prozesse gesetzt; per fork erben diese die Arrays
# ohne Kopie/Pickling (ein 500-MB-Bild nicht pro Prozess neu zu serialisieren).
_FEATURE_CTX = None


# (Zeilen-, Spalten-)Versatz der vier GLCM-Richtungen bei Abstand 1. Durch die
# Symmetrie (P + P.T) ist die Richtung bis aufs Vorzeichen egal.
_GLCM_OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1))


def glcm_texture_stats(patch):
    """(Kontrast, Homogenität, Energie, Korrelation) der GLCM eines uint8-Patches,
    Mittel über vier Richtungen bei Abstand 1, symmetrisch und normiert.

    Mathematisch identisch zu graycomatrix(levels=256, symmetric, normed) +
    graycoprops, aber nur über die tatsächlich vorkommenden Pixelpaare gerechnet
    statt über vier dichte 256x256-Matrizen: bei einem kleinen Segment sind das
    wenige Dutzend Paare, und graycoprops brauchte ~20 ms pro Segment (der
    größte Posten bei Hunderttausenden Segmenten). Abweichung nur im Rundungs-
    bereich (~1e-15 relativ)."""
    patch = patch.astype(np.int64)
    h, w = patch.shape
    stats = np.zeros((len(_GLCM_OFFSETS), 4))
    for k, (dr, dc) in enumerate(_GLCM_OFFSETS):
        a = patch[: h - dr, max(0, -dc): w - max(0, dc)].ravel()
        b = patch[dr:, max(0, dc): w - max(0, -dc)].ravel()
        i = np.concatenate([a, b])  # symmetrisch: jedes Paar in beide Richtungen
        j = np.concatenate([b, a])
        n = i.size
        diff = (i - j).astype(np.float64)
        contrast = float(np.mean(diff ** 2))
        homogeneity = float(np.mean(1.0 / (1.0 + diff ** 2)))
        _, counts = np.unique(i * 256 + j, return_counts=True)
        energy = float(np.sqrt(np.sum((counts / n) ** 2)))
        mean_i, mean_j = i.mean(), j.mean()
        std_i = float(np.sqrt(np.mean((i - mean_i) ** 2)))
        std_j = float(np.sqrt(np.mean((j - mean_j) ** 2)))
        if std_i < 1e-15 or std_j < 1e-15:
            correlation = 1.0  # wie graycoprops bei konstantem Patch
        else:
            correlation = float(np.mean((i - mean_i) * (j - mean_j)) / (std_i * std_j))
        stats[k] = (contrast, homogeneity, energy, correlation)
    return tuple(float(v) for v in stats.mean(axis=0))


def _build_feature_ctx(arr, segments, band_names, texture_band):
    """Einmal pro Mine vorberechnete Hilfsgrößen (Helligkeit, quantisiertes
    Texturband, Bounding Boxes) - von allen Teilstücken gemeinsam genutzt."""
    # Helligkeit für brightness_extreme_fraction: Mittel über die im Bild
    # vorhandenen sichtbaren Bänder. Fehlen alle drei (z.B. in Tests mit
    # einem minimalen Bandsatz), bleibt das Merkmal 0.0 statt abzustürzen.
    brightness_idx = [band_names.index(b) for b in ("blue", "green", "red") if b in band_names]
    if brightness_idx:
        with np.errstate(invalid="ignore"):
            # nodata-Bereiche sind über alle Bänder hinweg NaN -> "Mean of
            # empty slice" ist dort erwartet (wie bei northness/tpi weiter
            # oben), kein Hinweis auf ein Problem.
            brightness = np.nanmean(arr[..., brightness_idx], axis=-1)
    else:
        brightness = None

    tex_idx = band_names.index(texture_band)
    tex_band = arr[..., tex_idx]
    # für GLCM auf 8-Bit-Stufen quantisieren. Die echten Sentinel-2-Exporte
    # haben keinen expliziten NoData-Wert gesetzt, wolkenmaskierte Pixel
    # sind stattdessen direkt NaN -> ohne nan_to_num würde NaN nach uint8
    # gecastet (undefiniertes Ergebnis, nur eine RuntimeWarning als Hinweis),
    # was am Rand einzelner Segmente (Bounding-Box kann Nachbarpixel
    # außerhalb des eigentlichen Segments erwischen) zufällige "Textur"-
    # Werte in die GLCM-Berechnung einschleusen könnte.
    tex_band_q = np.nan_to_num(
        np.clip(
            ((tex_band - np.nanmin(tex_band)) /
             (np.nanmax(tex_band) - np.nanmin(tex_band) + 1e-6) * 255),
            0, 255,
        ),
        nan=0.0,
    ).astype(np.uint8)

    # Jedes Segment nur in SEINER Bounding Box betrachten (find_objects liefert
    # sie in einem Durchlauf). Früher wurde pro Segment `segments == seg_id` über
    # das GANZE Bild gerechnet - bei Hunderttausenden Segmenten auf Millionen
    # Pixeln quadratisch und der eigentliche Flaschenhals großer Minen. Die
    # Ergebnisse sind identisch, denn Pixel außerhalb der Bounding Box gehören
    # ohnehin nicht zum Segment.
    return {
        "arr": arr, "segments": segments, "band_names": band_names,
        "brightness": brightness, "tex_band_q": tex_band_q,
        "boxes": find_objects(segments.astype(np.int32)),
    }


def _segment_feature_rows(ctx, seg_ids):
    """Merkmalszeilen für die angegebenen Segment-IDs (unabhängig voneinander,
    deshalb beliebig auf Prozesse aufteilbar)."""
    arr, segments, band_names = ctx["arr"], ctx["segments"], ctx["band_names"]
    brightness, tex_band_q, boxes = ctx["brightness"], ctx["tex_band_q"], ctx["boxes"]
    rows = []
    for seg_id in seg_ids:
        box = boxes[int(seg_id) - 1]
        if box is None:
            continue
        m = segments[box] == seg_id
        if m.sum() < 4:
            continue  # zu kleines Segment, überspringen

        row = {"segment_id": int(seg_id), "n_pixels": int(m.sum())}

        for b, name in enumerate(band_names):
            vals = arr[box][..., b][m]
            row[f"{name}_mean"] = float(np.nanmean(vals))
            row[f"{name}_std"] = float(np.nanstd(vals))

        row["brightness_extreme_fraction"] = (
            brightness_extreme_fraction(brightness[box][m]) if brightness is not None else 0.0
        )

        # Bounding Box des Segments (einfache, robuste Näherung statt
        # exakter Segmentgeometrie - auch für die GLCM-Textur unten
        # verwendet).
        y0, y1, x0, x1 = box[0].start, box[0].stop, box[1].start, box[1].stop

        # Räumlicher Helligkeits-Split: max(|links-rechts|, |oben-unten|)
        # über die Bounding Box. Anders als brightness_extreme_fraction
        # (zählt nur den ANTEIL extrem heller/dunkler Pixel, egal wo sie
        # liegen) erfasst das gezielt einen SAUBEREN räumlichen Übergang
        # quer durchs Segment - die Signatur einer Grubenkante/eines
        # Hangs im Schlagschatten (eine Bildhälfte hell, die andere
        # dunkel), nicht die eines gleichmäßig verteilten Reifenhaufens.
        # An den echten Daten kalibriert (siehe README): bei >= 0.10
        # betrifft das 0 von 36 bekannten Dumps, fängt aber ~7% der
        # Hang-Falsch-Positiven (dem_slope_mean >= 5) ab.
        if brightness is not None:
            b_patch = brightness[y0:y1, x0:x1]
            bh, bw = b_patch.shape
            with np.errstate(invalid="ignore"):
                left, right = b_patch[:, : bw // 2], b_patch[:, bw // 2 :]
                top, bottom = b_patch[: bh // 2, :], b_patch[bh // 2 :, :]
                diffs = [
                    abs(np.nanmean(left) - np.nanmean(right)),
                    abs(np.nanmean(top) - np.nanmean(bottom)),
                ]
            row["spatial_split_contrast"] = (
                float(np.nanmax(diffs)) if not np.all(np.isnan(diffs)) else 0.0
            )
        else:
            row["spatial_split_contrast"] = 0.0

        # GLCM-Textur auf derselben Bounding Box
        patch = tex_band_q[y0:y1, x0:x1]
        if patch.shape[0] >= 2 and patch.shape[1] >= 2:
            (row["tex_contrast"], row["tex_homogeneity"],
             row["tex_energy"], row["tex_correlation"]) = glcm_texture_stats(patch)
        else:
            row["tex_contrast"] = row["tex_homogeneity"] = 0.0
            row["tex_energy"] = row["tex_correlation"] = 0.0

        rows.append(row)
    return rows


def _feature_chunk(seg_ids):
    return _segment_feature_rows(_FEATURE_CTX, seg_ids)


def compute_segment_features(arr, segments, band_names, texture_band,
                             n_workers=1, min_parallel_segments=PARALLEL_MIN_SEGMENTS):
    """Berechnet je Segment: Mittelwert & Std pro Band, Helligkeits-
    Bimodalität (siehe brightness_extreme_fraction()) und GLCM-Textur auf
    dem gewählten Band (z. B. Dark Surface Index).

    Bei sehr vielen Segmenten (große Minen) werden die Segmente auf n_workers
    Prozesse verteilt. Jedes Segment wird unabhängig berechnet, das Ergebnis
    ist daher identisch zur sequentiellen Variante (gleiche Reihenfolge)."""
    global _FEATURE_CTX
    seg_ids = np.unique(segments)
    seg_ids = seg_ids[seg_ids > 0]
    ctx = _build_feature_ctx(arr, segments, band_names, texture_band)

    use_pool = (
        n_workers and n_workers > 1 and len(seg_ids) >= min_parallel_segments
        and "fork" in multiprocessing.get_all_start_methods()
    )
    if use_pool:
        chunks = np.array_split(seg_ids, n_workers * 4)
        _FEATURE_CTX = ctx
        try:
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=multiprocessing.get_context("fork"),
                initializer=_init_worker_logging,
            ) as executor:
                rows = [r for part in executor.map(_feature_chunk, chunks) for r in part]
            return pd.DataFrame(rows)
        except (BrokenProcessPool, OSError, AssertionError) as exc:
            logger.warning(
                "Parallele Merkmalsberechnung nicht möglich (%s: %s) - rechne sequentiell.",
                type(exc).__name__, exc,
            )
        finally:
            _FEATURE_CTX = None

    return pd.DataFrame(_segment_feature_rows(ctx, seg_ids))


# ------------------------------------------------------------------
# 4) Labeling anhand der digitalisierten Dump-Polygone
# ------------------------------------------------------------------
def label_segments(seg_gdf, labels_gdf, min_overlap_ratio):
    """Markiert ein Segment als positiv (label=1), wenn der Anteil
    seiner Fläche, der mit einem Dump-Label überlappt, >= min_overlap_ratio."""
    seg_gdf = seg_gdf.copy()
    if "shape_area" not in seg_gdf.columns:
        seg_gdf["shape_area"] = seg_gdf.geometry.area

    if labels_gdf.empty:
        seg_gdf["overlap_area"] = 0.0
        seg_gdf["overlap_ratio"] = 0.0
        seg_gdf["label"] = 0
        logger.info("Keine Dump-Labels für diese Mine vorhanden -> alle Segmente label=0.")
        return seg_gdf

    overlay = gpd.overlay(seg_gdf, labels_gdf[["geometry"]], how="intersection")
    overlay["overlap_area"] = overlay.geometry.area
    overlap_by_seg = overlay.groupby(["mine_id", "segment_id"])["overlap_area"].sum()

    seg_gdf = seg_gdf.set_index(["mine_id", "segment_id"])
    seg_gdf["overlap_area"] = overlap_by_seg.reindex(seg_gdf.index).fillna(0.0)
    seg_gdf["overlap_ratio"] = seg_gdf["overlap_area"] / seg_gdf["shape_area"]
    seg_gdf["label"] = (seg_gdf["overlap_ratio"] >= min_overlap_ratio).astype(int)
    seg_gdf = seg_gdf.reset_index()

    n_pos = int(seg_gdf["label"].sum())
    if n_pos == 0:
        logger.warning(
            "Dump-Labels für diese Mine vorhanden, aber kein Segment erreicht "
            "min_overlap_ratio=%.2f -> 0 positive Segmente. Falls das unerwartet "
            "ist, prüfe ob die mine_id in labels_path wirklich zu dieser Mine passt.",
            min_overlap_ratio,
        )
    return seg_gdf


def _init_worker_logging():
    """Wird einmal pro Worker-Prozess ausgeführt. Bei 'fork' (Linux-Default)
    ist das Logging meist schon aus dem Hauptprozess geerbt, bei 'spawn'
    (Windows/macOS-Default) startet der Prozess aber komplett neu und hätte
    sonst keine Ausgabe. basicConfig() ist ein No-op, falls bereits
    konfiguriert, schadet also in keinem der beiden Fälle."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def filter_to_mine_boundary(seg_polys, mine_boundary_gdf, mine_id, buffer_m=0.0):
    """Entfernt Segmente, die außerhalb der tatsächlichen Minen-Grenze liegen.

    Beim Sentinel-2-Export wurde um jede Mine ein 500m-Puffer gelegt (siehe
    gee/sentinel2_export.js), sodass jedes Bild auch einen erheblichen Teil
    umliegendes Gelände enthält, das nie eine Reifenhalde enthalten kann -
    Berge, Wüste, etc. Fälschlicherweise als positiv erkannte Segmente in
    diesem Randbereich lassen sich damit von vornherein ausschließen, statt
    im Nachhinein herausgefiltert zu werden.

    Wenn keine Grenze für diese Mine gefunden wird, bleibt seg_polys
    unverändert (kein Fehler) - die Grenzdatei ist optional."""
    if mine_boundary_gdf is None or mine_boundary_gdf.empty:
        return seg_polys

    boundary = mine_boundary_gdf[mine_boundary_gdf["_mine_id_str"] == str(mine_id).strip()]
    if boundary.empty:
        return seg_polys

    if boundary.crs != seg_polys.crs:
        boundary = boundary.to_crs(seg_polys.crs)

    boundary_geom = boundary.geometry.union_all()
    if buffer_m:
        boundary_geom = boundary_geom.buffer(buffer_m)

    n_before = len(seg_polys)
    filtered = seg_polys[seg_polys.intersects(boundary_geom)].copy()
    n_dropped = n_before - len(filtered)
    if n_dropped > 0:
        logger.info(
            "  %s: %d von %d Segmenten liegen außerhalb der Minen-Grenze "
            "(+%.0fm Puffer) und werden ausgeschlossen.",
            mine_id, n_dropped, n_before, buffer_m,
        )
    return filtered


def _read_aligned(s2_path, directory, shape, n_bands, what, fetch_script):
    """Liest die Zusatzdatei (gleicher Dateiname wie das Sentinel-2-Bild) aus
    directory und prüft, dass sie exakt dasselbe Pixelraster hat. Rückgabe:
    (Bänder (n, H, W), Pixelgröße x, Pixelgröße y, CRS, Transform), oder None
    mit Warnung, wenn die Datei fehlt - der Aufrufer liefert dann NaN-Bänder,
    damit alle Minen dieselben Merkmalsspalten haben."""
    path = os.path.join(directory, os.path.basename(s2_path))
    if not os.path.exists(path):
        logger.warning(
            "Keine %s-Datei für %s in %s gefunden -> diese Merkmale sind für die "
            "Mine leer. Mit src/%s nachladen.",
            what, os.path.basename(s2_path), directory, fetch_script,
        )
        return None
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        px_x, px_y = abs(src.transform.a), abs(src.transform.e)
        crs, transform = src.crs, src.transform
    if data.shape[0] != n_bands or data.shape[1:] != tuple(shape):
        raise ValueError(
            f"{path}: erwartet {n_bands} Band/Bänder im Raster {tuple(shape)} (wie "
            f"das Sentinel-2-Bild), gefunden {data.shape[0]} im Raster "
            f"{data.shape[1:]}. Datei mit {fetch_script} --overwrite neu erzeugen."
        )
    return data, px_x, px_y, crs, transform


def load_s1_for_mine(s2_path, s1_dir, shape):
    """Sentinel-1 als (H, W, 3): VV, VH und VH-VV (alles dB)."""
    read = _read_aligned(s2_path, s1_dir, shape, 2, "Sentinel-1", "fetch_sentinel1.py")
    if read is None:
        return np.full((*shape, len(S1_BAND_NAMES)), np.nan, dtype=np.float32)
    data, *_ = read
    vv, vh = data
    return np.stack([vv, vh, vh - vv], axis=-1)


def load_s2_temporal_for_mine(s2_path, s2t_dir, shape):
    """Zeitreihen-Merkmale aus mehreren Sentinel-2-Aufnahmen als (H, W, 2),
    siehe fetch_s2_timeseries.py."""
    read = _read_aligned(s2_path, s2t_dir, shape, len(S2T_BAND_NAMES),
                         "Sentinel-2-Zeitreihen", "fetch_s2_timeseries.py")
    if read is None:
        return np.full((*shape, len(S2T_BAND_NAMES)), np.nan, dtype=np.float32)
    data, *_ = read
    return np.moveaxis(data, 0, -1)


def select_s2my_names(cfg):
    """Welche der sechs Mehrjahres-Merkmale genutzt werden (cfg["s2my_bands"], Komma-Liste
    oder Liste; leer = alle). Zum Testen, welche davon etwas bringen."""
    wanted = cfg.get("s2my_bands")
    if not wanted:
        return list(S2MY_BAND_NAMES)
    names = [n.strip() for n in wanted.split(",")] if isinstance(wanted, str) else list(wanted)
    unknown = [n for n in names if n not in S2MY_BAND_NAMES]
    if unknown or not names:
        raise ValueError(f"Unbekannte Mehrjahres-Merkmale {unknown}; erlaubt: {S2MY_BAND_NAMES}")
    return [n for n in S2MY_BAND_NAMES if n in names]      # feste Reihenfolge


def load_s2_multiyear_for_mine(s2_path, s2my_dir, shape, names=None):
    """Mehrjahres-Merkmale aus Sentinel-2 als (H, W, n), siehe fetch_s2_multiyear.py.
    names: Auswahl aus S2MY_BAND_NAMES (Standard: alle sechs)."""
    names = list(names) if names else list(S2MY_BAND_NAMES)
    cols = [S2MY_BAND_NAMES.index(n) for n in names]
    read = _read_aligned(s2_path, s2my_dir, shape, len(S2MY_BAND_NAMES),
                         "Sentinel-2-Mehrjahres", "fetch_s2_multiyear.py")
    if read is None:
        return np.full((*shape, len(names)), np.nan, dtype=np.float32)
    data, *_ = read
    return np.moveaxis(data[cols], 0, -1)


def load_s1_temporal_for_mine(s2_path, s1t_dir, shape):
    """Radar-Zeitreihen-Merkmale aus mehreren Sentinel-1-Aufnahmen als
    (H, W, 2), siehe fetch_sentinel1_timeseries.py."""
    read = _read_aligned(s2_path, s1t_dir, shape, len(S1T_BAND_NAMES),
                         "Sentinel-1-Zeitreihen", "fetch_sentinel1_timeseries.py")
    if read is None:
        return np.full((*shape, len(S1T_BAND_NAMES)), np.nan, dtype=np.float32)
    data, *_ = read
    return np.moveaxis(data, 0, -1)


def solar_position(lat, lon, when):
    """Sonnenazimut (im Uhrzeigersinn ab Norden, 0-360°) und Sonnenhöhe über
    dem Horizont (Grad) für einen Ort und Zeitpunkt. when: ISO-Zeitstempel
    oder Timestamp, UTC. Nutzt pvlib für eine astronomisch korrekte
    Berechnung (Sonnenposition, nicht nur eine grobe Näherung)."""
    result = pvlib.solarposition.get_solarposition(pd.DatetimeIndex([when]), lat, lon)
    return float(result["azimuth"].iloc[0]), float(result["apparent_elevation"].iloc[0])


def hillshade(slope_deg, aspect_deg, sun_azimuth_deg, sun_elevation_deg):
    """Erwartete Beleuchtungsstärke einer geneigten Fläche bei gegebenem
    Sonnenstand (Standard-Hillshade-Formel, Skalarprodukt aus Flächen-
    normale und Sonnenrichtung): 1 = Fläche zeigt direkt zur Sonne, 0 =
    Sonnenstrahlen streifen die Fläche im rechten Winkel, negativ = Fläche
    zeigt von der Sonne weg (Eigenverschattung). Erfasst NUR die
    Ausrichtung der Fläche selbst, KEINEN Schattenwurf durch umliegendes,
    höheres Gelände (dafür wäre Raytracing über das gesamte DEM nötig)."""
    slope = np.radians(slope_deg)
    aspect = np.radians(aspect_deg)
    zenith = np.radians(90.0 - sun_elevation_deg)
    azimuth = np.radians(sun_azimuth_deg)
    return (
        np.cos(zenith) * np.cos(slope)
        + np.sin(zenith) * np.sin(slope) * np.cos(azimuth - aspect)
    )


def terrain_features(elevation, pixel_size_x, pixel_size_y, tpi_window_px, lat=-24.0, lon=-69.5):
    """Leitet aus der Höhe (H, W) fünf Gelände-Merkmale ab, Rückgabe (H, W, 5):

    - dem_slope: Hangneigung in Grad (0 = flach)
    - dem_tpi: Höhe minus mittlere Höhe der Umgebung (tpi_window_px) in m;
      positiv = Kuppe/Grat, negativ = Senke/Tal
    - dem_northness: cos(Hangausrichtung) * sin(Neigung), -1..1. Positiv =
      nach Norden geneigt. Auf der Südhalbkugel steht die Sonne im Norden,
      nach Süden geneigte Hänge (negativ) liegen also im Schatten. Flaches
      Gelände = 0, egal wohin es minimal geneigt ist.
    - dem_hillshade_dec / dem_hillshade_jun: siehe hillshade(), ausgewertet
      am lokalen Mittag zur Süd-Sommer- bzw. Süd-Wintersonnenwende (siehe
      SOLSTICE_DATETIMES). Genauer als dem_northness, weil es den echten
      Sonnenstand nutzt statt nur "Nord vs. Süd" - wichtig z.B. in engen
      Tälern, wo ein Hang je nach Jahreszeit unterschiedlich stark
      beschattet ist, was dem_northness als Mittelwert über das Segment
      teils verwässert.

    lat/lon (WGS84) bestimmen den Sonnenstand für die Hillshade-Merkmale und
    sollten die tatsächliche Lage der Mine sein (siehe load_dem_for_mine());
    der Standardwert ist nur eine grobe Näherung für Tests.

    Erwartet ein nordausgerichtetes Raster (Zeilen laufen nach Süden)."""
    elev = elevation.astype(np.float64)
    missing = ~np.isfinite(elev)
    if missing.all():
        return np.full((*elev.shape, len(DEM_BAND_NAMES)), np.nan, dtype=np.float32)
    if missing.any():
        elev = np.where(missing, np.nanmean(elev), elev)

    dz_drow, dz_dcol = np.gradient(elev, pixel_size_y, pixel_size_x)
    grad = np.hypot(dz_drow, dz_dcol)
    slope = np.degrees(np.arctan(grad))
    # Zeilen laufen nach Süden: dz_drow > 0 heißt, das Gelände fällt nach
    # Norden ab -> Hang zeigt nach Norden.
    with np.errstate(invalid="ignore", divide="ignore"):
        northness = np.where(grad > 0, dz_drow / grad, 0.0) * np.sin(np.arctan(grad))
    tpi = elev - uniform_filter(elev, size=tpi_window_px, mode="nearest")

    # Hangausrichtung im Uhrzeigersinn ab Norden (0-360°), aus denselben
    # Gradienten wie dem_northness - siehe deren Vorzeichenerklärung oben.
    aspect = np.degrees(np.arctan2(-dz_dcol, dz_drow)) % 360

    hillshades = []
    for season in ("dec", "jun"):
        sun_azimuth, sun_elevation = solar_position(lat, lon, SOLSTICE_DATETIMES[season])
        hillshades.append(hillshade(slope, aspect, sun_azimuth, sun_elevation))

    out = np.stack([slope, tpi, northness, *hillshades], axis=-1).astype(np.float32)
    out[missing] = np.nan
    return out


def load_dem_for_mine(s2_path, dem_dir, shape, tpi_window_px):
    """Höhenmodell -> Gelände-Merkmale (H, W, 5), siehe terrain_features().
    Die Lage der Mine (für den Sonnenstand der Hillshade-Merkmale) wird aus
    der Bildmitte des Höhenmodell-Rasters bestimmt."""
    read = _read_aligned(s2_path, dem_dir, shape, 1, "Höhenmodell", "fetch_dem.py")
    if read is None:
        return np.full((*shape, len(DEM_BAND_NAMES)), np.nan, dtype=np.float32)
    data, px_x, px_y, crs, transform = read
    height, width = data.shape[1:]
    center_x, center_y = transform @ (width / 2.0, height / 2.0)
    lon, lat = warp_transform(crs, "EPSG:4326", [center_x], [center_y])
    return terrain_features(data[0], px_x, px_y, tpi_window_px, lat=lat[0], lon=lon[0])


def _process_one_mine(path, cfg, labels_gdf, target_crs, mine_boundaries_gdf=None):
    """Verarbeitet eine einzelne Mine vollständig (Segmentierung, Merkmale,
    Labeling). Eigene Top-Level-Funktion, damit sie in build_dataset()
    parallel in mehreren Prozessen laufen kann - jede Mine ist unabhängig
    von jeder anderen, es gibt also keinen Grund, sie nacheinander
    abzuarbeiten."""
    mine_id = extract_mine_id(path)
    logger.info("Verarbeite Mine '%s' (%s) ...", mine_id, os.path.basename(path))

    arr, transform, crs, nodata = load_mine_raster(
        path, expected_n_bands=len(cfg["band_names"])
    )

    if crs != target_crs:
        logger.warning(
            "Mine '%s' hat ein anderes Koordinatensystem (%s) als die erste "
            "Mine (%s). Segmente werden nach %s reprojiziert, damit beim "
            "Zusammenführen aller Minen nichts verschoben wird.",
            mine_id, crs, target_crs, target_crs,
        )

    segments = segment_mine(
        arr, cfg["n_segments_per_mine"], cfg["compactness"], nodata=nodata,
        target_segment_px=cfg.get("target_segment_px"),
        max_segments=cfg.get("max_segments_per_mine"),
    )
    seg_polys = segments_to_polygons(segments, transform, crs, mine_id)
    seg_polys = compute_shape_features(seg_polys)
    seg_polys = filter_to_mine_boundary(
        seg_polys, mine_boundaries_gdf, mine_id,
        buffer_m=cfg.get("mine_boundary_buffer_m", 0.0),
    )

    mine_labels = labels_gdf[labels_gdf["_mine_id_str"] == str(mine_id).strip()]
    if not labels_gdf.empty and mine_labels.empty:
        logger.warning(
            "Keine Einträge in %s mit %s == '%s' gefunden (insgesamt %d "
            "Label-Einträge vorhanden). Prüfe, ob die aus dem Dateinamen "
            "extrahierte mine_id zum Namensschema in labels_path passt.",
            cfg["labels_path"], cfg["mine_id_field"], mine_id, len(labels_gdf),
        )
    if not mine_labels.empty and mine_labels.crs != crs:
        mine_labels = mine_labels.to_crs(crs)

    seg_polys = label_segments(seg_polys, mine_labels, cfg["min_overlap_ratio"])

    if crs != target_crs:
        seg_polys = seg_polys.to_crs(target_crs)

    # Zusatzquellen erst NACH der Segmentierung anhängen: die Segmente kommen
    # weiter nur aus dem Sentinel-2-Bild, die übrigen Quellen liefern nur
    # zusätzliche Merkmale.
    extra_arrays, feat_band_names = [], list(cfg["band_names"])
    shape = arr.shape[:2]
    if cfg.get("s1_dir"):
        extra_arrays.append(load_s1_for_mine(path, cfg["s1_dir"], shape))
        feat_band_names += S1_BAND_NAMES
    if cfg.get("dem_dir"):
        extra_arrays.append(load_dem_for_mine(
            path, cfg["dem_dir"], shape, cfg.get("dem_tpi_window_px", 31)
        ))
        feat_band_names += DEM_BAND_NAMES
    if cfg.get("s2t_dir"):
        extra_arrays.append(load_s2_temporal_for_mine(path, cfg["s2t_dir"], shape))
        feat_band_names += S2T_BAND_NAMES
    if cfg.get("s2my_dir"):
        s2my_names = select_s2my_names(cfg)
        extra_arrays.append(load_s2_multiyear_for_mine(path, cfg["s2my_dir"], shape, s2my_names))
        feat_band_names += s2my_names
    if cfg.get("s1t_dir"):
        extra_arrays.append(load_s1_temporal_for_mine(path, cfg["s1t_dir"], shape))
        feat_band_names += S1T_BAND_NAMES
    feat_arr = np.concatenate([arr, *extra_arrays], axis=-1) if extra_arrays else arr

    feats = compute_segment_features(
        feat_arr, segments, feat_band_names, cfg["texture_band"],
        n_workers=cfg.get("feature_workers") or os.cpu_count() or 1,
    )
    merged = seg_polys.merge(feats, on="segment_id", how="inner")

    n_pos = int(merged["label"].sum())
    logger.info("  %s: %d Segmente, davon %d positiv", mine_id, len(merged), n_pos)

    features = pd.DataFrame(merged.drop(columns="geometry"))
    polygons = gpd.GeoDataFrame(
        merged[["mine_id", "segment_id", "geometry"]].copy(), crs=target_crs
    )
    return features, polygons


# ------------------------------------------------------------------
# 5) Gesamten Datensatz aus allen Minen aufbauen
# ------------------------------------------------------------------
def build_dataset(cfg):
    labels_gdf = gpd.read_file(cfg["labels_path"])
    if cfg["mine_id_field"] not in labels_gdf.columns:
        raise ValueError(
            f"Spalte '{cfg['mine_id_field']}' (CONFIG['mine_id_field']) nicht in "
            f"{cfg['labels_path']} gefunden. Vorhandene Spalten: {list(labels_gdf.columns)}"
        )
    # mine_id robust als String vergleichen: extract_mine_id() liefert immer
    # einen String, die Label-Spalte kann je nach Quelldatei int oder str sein.
    # Ohne diese Normalisierung matcht z.B. mine_id "12" (aus Dateiname) nicht
    # gegen den int 12 in labels_path, und die Mine bekommt still label=0.
    labels_gdf = labels_gdf.copy()
    labels_gdf["_mine_id_str"] = labels_gdf[cfg["mine_id_field"]].astype(str).str.strip()

    # Minen-Grenzen sind optional: nur gesetzt, wenn mine_boundary_path in
    # der Config einen Wert hat. Ohne diese Datei läuft die Pipeline exakt
    # wie zuvor (kein Filtern).
    mine_boundaries_gdf = None
    boundary_path = cfg.get("mine_boundary_path")
    if boundary_path:
        if not os.path.exists(boundary_path):
            raise FileNotFoundError(
                f"mine_boundary_path='{boundary_path}' (CONFIG['mine_boundary_path']) "
                "nicht gefunden."
            )
        mine_boundaries_gdf = gpd.read_file(boundary_path)
        boundary_id_field = cfg.get("mine_boundary_id_field", "mine_id")
        if boundary_id_field not in mine_boundaries_gdf.columns:
            raise ValueError(
                f"Spalte '{boundary_id_field}' (CONFIG['mine_boundary_id_field']) nicht "
                f"in {boundary_path} gefunden. Vorhandene Spalten: "
                f"{list(mine_boundaries_gdf.columns)}"
            )
        mine_boundaries_gdf = mine_boundaries_gdf.copy()
        mine_boundaries_gdf["_mine_id_str"] = (
            mine_boundaries_gdf[boundary_id_field].astype(str).str.strip()
        )
        logger.info(
            "%d Minen-Grenzen aus %s geladen (Puffer: %.0fm).",
            len(mine_boundaries_gdf), boundary_path, cfg.get("mine_boundary_buffer_m", 0.0),
        )

    # Größte Dateien zuerst: Dateigröße korreliert stark mit Verarbeitungszeit
    # (mehr Pixel -> mehr Segmentierung/Merkmalsberechnung), und jede Mine
    # läuft einzeln single-threaded (siehe Modul-Docstring). Fängt man mit den
    # größten Minen an, laufen sie von Anfang an auf eigenen Prozessen, statt
    # erst spät in der Warteschlange zu starten und die Gesamtlaufzeit ganz
    # am Ende in die Länge zu ziehen ("Longest Processing Time first", ein
    # Standard-Scheduling-Trick für Jobs mit stark unterschiedlicher Größe
    # bei fester Anzahl Worker).
    imagery_paths = sorted(
        glob.glob(os.path.join(cfg["imagery_dir"], "*.tif")),
        key=os.path.getsize, reverse=True,
    )
    if not imagery_paths:
        raise FileNotFoundError(f"Keine GeoTIFFs in {cfg['imagery_dir']} gefunden.")
    logger.info("%d GeoTIFFs in %s gefunden.", len(imagery_paths), cfg["imagery_dir"])

    # Ziel-Koordinatensystem = das der ersten Mine (nur den Header lesen, nicht
    # die Bilddaten). Muss VOR der parallelen Verarbeitung feststehen, weil bei
    # paralleler Ausführung nicht mehr garantiert ist, welche Mine "zuerst"
    # fertig wird.
    with rasterio.open(imagery_paths[0]) as src:
        target_crs = src.crs

    n_jobs = cfg.get("n_jobs") or os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, len(imagery_paths)))

    # Ein Fehler bei EINER Mine (kaputte Datei, Speicherproblem, ...) darf
    # nicht die Ergebnisse aller anderen, bereits erfolgreich verarbeiteten
    # Minen wegwerfen - das kann bei 138 Minen sehr teuer werden. Fehler
    # werden daher pro Mine abgefangen und geloggt, die Verarbeitung läuft
    # für die übrigen Minen weiter.
    results = [None] * len(imagery_paths)
    failed = []
    if n_jobs <= 1:
        for i, path in enumerate(imagery_paths):
            try:
                results[i] = _process_one_mine(
                    path, cfg, labels_gdf, target_crs, mine_boundaries_gdf
                )
            except Exception:
                logger.exception("Mine '%s' fehlgeschlagen, wird übersprungen.", path)
                failed.append(path)
    else:
        logger.info(
            "Verarbeite %d Minen parallel mit %d Prozessen ...",
            len(imagery_paths), n_jobs,
        )
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker_logging) as executor:
            future_to_idx = {
                executor.submit(
                    _process_one_mine, path, cfg, labels_gdf, target_crs, mine_boundaries_gdf
                ): i
                for i, path in enumerate(imagery_paths)
            }
            crashed = []
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except BrokenProcessPool:
                    # Ein Worker wurde hart beendet (meist Speichermangel). Das reißt
                    # ALLE noch laufenden/wartenden Minen des Pools mit - sie werden
                    # unten einzeln und isoliert wiederholt.
                    crashed.append(idx)
                except Exception:
                    logger.exception(
                        "Mine '%s' fehlgeschlagen, wird übersprungen.", imagery_paths[idx]
                    )
                    failed.append(imagery_paths[idx])
        if crashed:
            logger.warning(
                "Ein Worker-Prozess wurde abrupt beendet (vermutlich Speichermangel); "
                "wiederhole %d betroffene Minen einzeln: %s",
                len(crashed), [os.path.basename(imagery_paths[i]) for i in crashed],
            )
            for idx in sorted(crashed, key=lambda i: -os.path.getsize(imagery_paths[i])):
                try:
                    with ProcessPoolExecutor(max_workers=1, initializer=_init_worker_logging) as solo:
                        results[idx] = solo.submit(
                            _process_one_mine, imagery_paths[idx], cfg, labels_gdf,
                            target_crs, mine_boundaries_gdf,
                        ).result()
                except Exception:
                    logger.exception(
                        "Mine '%s' fehlgeschlagen (auch einzeln), wird übersprungen.", imagery_paths[idx]
                    )
                    failed.append(imagery_paths[idx])

    if failed:
        logger.warning(
            "%d von %d Minen fehlgeschlagen und übersprungen: %s",
            len(failed), len(imagery_paths), [os.path.basename(p) for p in failed],
        )
    succeeded = [r for r in results if r is not None]
    if not succeeded:
        raise RuntimeError("Alle Minen sind fehlgeschlagen, kein Segment-Datensatz erzeugt.")

    all_features = [feats for feats, _ in succeeded]
    all_polygons = [polys for _, polys in succeeded]

    feature_df = pd.concat(all_features, ignore_index=True)
    polygons_gdf = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True), crs=target_crs)
    return feature_df, polygons_gdf


def _build_classifier(cfg):
    """Baut den Random-Forest-Klassifikator. Nutzt BalancedRandomForestClassifier
    (imbalanced-learn) statt eines gewöhnlichen RandomForestClassifier mit
    class_weight="balanced": bei den echten Daten sind nur 36 von 174.663
    Segmenten positiv (~1:4850). class_weight reduziert lediglich das
    Gewicht der Mehrheitsklasse im Trainings-Loss, ändert aber nichts am
    Bootstrap-Sample jedes einzelnen Baums - bei so extremer Schieflage
    bekommt praktisch jeder Baum kaum je ein positives Beispiel zu sehen.
    Getestet an den echten Daten: class_weight="balanced" fand 0 von 9
    bekannten Dumps (Out-of-Fold), AUC=0.585. BalancedRandomForestClassifier
    zieht pro Baum ein tatsächlich ausgeglichenes Sample und fand 7-9 von 9,
    AUC=0.894 - derselbe Datensatz, nur anderes Resampling."""
    return BalancedRandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        random_state=cfg["random_state"],
        n_jobs=-1,
    )


# ------------------------------------------------------------------
# 6) Training + räumliche Validierung (Split NACH Mine, wie im Report)
# ------------------------------------------------------------------
def train_and_evaluate(feature_df, polygons_gdf, cfg, hard_neg_mask=None, hard_neg_factor=1):
    # overlap_area/overlap_ratio werden direkt aus dem Label abgeleitet
    # (Data Leakage) und dürfen daher NICHT als Merkmal verwendet werden.
    # shape_area/shape_perimeter/shape_compactness sind dagegen legitime
    # Formmerkmale und bleiben bewusst drin.
    non_feature_cols = ("segment_id", "mine_id", "label", "overlap_area", "overlap_ratio")
    feature_cols = [c for c in feature_df.columns if c not in non_feature_cols]
    logger.info("Verwende %d Merkmale: %s", len(feature_cols), feature_cols)

    X = feature_df[feature_cols].fillna(0.0).values
    y = feature_df["label"].values
    groups = feature_df["mine_id"].values
    hard = np.asarray(hard_neg_mask, dtype=bool) if hard_neg_mask is not None else np.zeros(len(y), dtype=bool)

    def _augment(idx):
        """Harte Negative im Training (nur dort!) k-mal wiederholen - Out-of-Fold-Vorhersagen
        bleiben unverändert eine Zeile je Segment."""
        if hard_neg_factor <= 1 or not hard[idx].any():
            return idx
        return np.concatenate([idx] + [idx[hard[idx]]] * (int(hard_neg_factor) - 1))

    n_mines = feature_df["mine_id"].nunique()
    n_splits = min(5, n_mines)
    if n_splits < 5:
        logger.warning(
            "Nur %d Minen im Datensatz -> GroupKFold läuft mit n_splits=%d statt 5.",
            n_mines, n_splits,
        )
    gkf = GroupKFold(n_splits=n_splits)

    # Sammelt für jedes Segment die Vorhersage AUS DER FOLD, in der seine
    # Mine im Validierungs-Set war (nie aus dem Training gesehen). Das ist
    # die einzige ehrliche Grundlage für eine Mine-Level-Erkennungsquote -
    # das später auf ALLEN Daten trainierte final_clf hat jede Mine schon
    # gesehen und würde eine völlig irreführende ~100%-Trefferquote zeigen.
    out_of_fold_proba = np.full(len(y), np.nan)
    threshold = cfg.get("classification_threshold", 0.5)

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        val_mines = sorted(set(groups[val_idx]))
        train_classes = np.unique(y[train_idx])
        if len(train_classes) < 2:
            # Trainings-Split dieser Fold enthält nur eine Klasse (z.B. eine
            # Mine ganz ohne positive Segmente). BalancedRandomForestClassifier
            # wirft dabei schon beim fit() einen Fehler (anders als ein
            # gewöhnlicher RandomForestClassifier, der einfach ein
            # Ein-Klassen-Modell bauen würde) - also gar nicht erst fitten.
            only_class = float(train_classes[0])
            proba = np.full(len(val_idx), only_class)
            logger.warning(
                "Fold %d (Validierungs-Minen: %s): Trainingsdaten enthalten nur "
                "Klasse %.0f -> Modell kann nicht lernen, verwende konstante "
                "proba=%.1f für diese Fold.", fold, val_mines, only_class, only_class,
            )
        else:
            clf = _build_classifier(cfg)
            aug = _augment(train_idx)
            clf.fit(X[aug], y[aug])
            proba = clf.predict_proba(X[val_idx])[:, 1]
        pred = (proba >= threshold).astype(int)
        out_of_fold_proba[val_idx] = proba

        f1 = f1_score(y[val_idx], pred, zero_division=0)
        try:
            auc = roc_auc_score(y[val_idx], proba)
        except ValueError:
            auc = float("nan")  # falls eine Fold nur eine Klasse enthält
            logger.warning(
                "Fold %d (Validierungs-Minen: %s) enthält nur eine Klasse -> AUC=nan.",
                fold, val_mines,
            )

        fold_scores.append({"fold": fold, "f1": f1, "auc": auc, "val_mines": val_mines})
        logger.info("  Fold %d (Minen=%s): F1=%.3f  AUC=%.3f", fold, val_mines, f1, auc)

    scores_df = pd.DataFrame(fold_scores)
    logger.info("Mittelwerte über alle Folds:\n%s", scores_df[["f1", "auc"]].mean())

    # Ehrliche Mine-Level-Erkennungsquote NUR aus Out-of-Fold-Vorhersagen
    # (jedes Segment wurde von einem Modell vorhergesagt, das seine Mine nie
    # im Training gesehen hat). Wichtig: das ist eine andere Zahl als die,
    # die predict_and_export() später für das QGIS-Export-GeoPackage
    # berechnet - jene nutzt das finale Modell, das auf ALLEN Minen
    # trainiert wurde, und ist daher KEINE Generalisierungs-Schätzung.
    require_neighbor_below = cfg.get("require_neighbor_below")
    require_compact_cluster_below = cfg.get("require_compact_cluster_below")
    min_cluster_compactness = cfg.get("min_cluster_compactness", 0.15)
    oof_result = feature_df[["mine_id", "segment_id", "label"]].copy()
    oof_result["dump_proba"] = out_of_fold_proba
    oof_result = oof_result.merge(
        polygons_gdf[["mine_id", "segment_id", "geometry"]], on=["mine_id", "segment_id"], how="left"
    )
    oof_result = gpd.GeoDataFrame(oof_result, geometry="geometry", crs=polygons_gdf.crs)
    oof_result["dump_pred"] = (oof_result["dump_proba"] >= threshold).astype(int)
    if cfg.get("output_dir"):
        os.makedirs(cfg["output_dir"], exist_ok=True)
        oof_result[["mine_id", "segment_id", "label", "dump_proba"]].to_pickle(
            os.path.join(cfg["output_dir"], "oof_predictions.pkl"))
    if require_neighbor_below is not None:
        oof_result = apply_neighbor_filter(oof_result, high_confidence=require_neighbor_below)
    if require_compact_cluster_below is not None:
        oof_result = apply_cluster_shape_filter(
            oof_result, high_confidence=require_compact_cluster_below,
            min_compactness=min_cluster_compactness,
        )
    logger.info(
        "Ehrliche (Out-of-Fold-) Mine-Level-Erkennung bei Schwellwert=%.2f, "
        "NUR aus Kreuzvalidierung, jede Mine wurde von einem Modell "
        "vorhergesagt, das sie nie im Training gesehen hat:", threshold,
    )
    summarize_mine_detection(oof_result)
    report_threshold_sweep(
        oof_result, require_neighbor_below=require_neighbor_below,
        require_compact_cluster_below=require_compact_cluster_below,
        min_cluster_compactness=min_cluster_compactness,
    )

    # finales Modell auf ALLEN Daten für die spätere Vollprädiktion
    if len(np.unique(y)) < 2:
        # Der GESAMTE Datensatz enthält nur eine Klasse (z.B. 0 positive
        # Segmente über alle Minen hinweg). BalancedRandomForestClassifier
        # wirft dabei schon beim fit() einen Fehler; ein gewöhnlicher
        # RandomForestClassifier baut dagegen anstandslos ein
        # Ein-Klassen-Modell, das predict_and_export() bereits über seine
        # eigene len(classes_)<2-Prüfung sicher behandelt.
        final_clf = RandomForestClassifier(
            n_estimators=1, random_state=cfg["random_state"]
        )
    else:
        final_clf = _build_classifier(cfg)
    aug_all = _augment(np.arange(len(y)))
    final_clf.fit(X[aug_all], y[aug_all])

    importances = pd.Series(final_clf.feature_importances_, index=feature_cols)
    logger.info(
        "Wichtigste Merkmale:\n%s", importances.sort_values(ascending=False).head(10)
    )

    return final_clf, feature_cols, scores_df


def compute_is_building(result, osm_buildings_dir):
    """Fügt eine rein informative is_building-Spalte hinzu (True/False):
    Segmente, die ein OSM-Gebäude-Polygon (building=*, siehe
    fetch_osm_buildings.py) überlappen, ODER ein solches Segment berühren.

    Ändert dump_pred/dump_proba NICHT - anders als apply_neighbor_filter()
    und apply_cluster_shape_filter() ist das kein Filter, sondern eine
    zusätzliche Spalte zum selbst Ein-/Ausblenden in QGIS (Filter/Query
    Builder auf "is_building" = 0), weil die OSM-Gebäudeabdeckung je nach
    Mine stark unterschiedlich vollständig ist.

    An den echten Daten validiert (siehe README, "Investigated: separating
    buildings from dumps"): betrifft 0 von 36 bekannten Dumps (auch nicht
    über die Nachbarschafts-Ausweitung), fängt aber im Schnitt nur ~1.6%
    der grenzwertigen Falsch-Positiven ab - an gut kartierten Standorten
    (einzelne Minen-Anlagen) deutlich mehr (~20-30%), an den meisten
    anderen praktisch nichts, weil dort kaum OSM-Gebäude erfasst sind.

    Erwartet result mit den Spalten mine_id und geometry. Für Minen ohne
    passende <mine_id>.gpkg in osm_buildings_dir bleibt is_building=False
    (kein Fehler - die OSM-Daten sind optional)."""
    result = result.copy()
    result["is_building"] = False
    if not osm_buildings_dir:
        return result

    for mine_id, group in result.groupby("mine_id"):
        building_path = os.path.join(osm_buildings_dir, f"{mine_id}.gpkg")
        if not os.path.exists(building_path):
            continue
        buildings = gpd.read_file(building_path)
        if buildings.empty:
            continue
        if buildings.crs != group.crs:
            buildings = buildings.to_crs(group.crs)
        building_union = buildings.geometry.union_all()

        direct_idx = group.index[group.geometry.intersects(building_union)]
        if len(direct_idx) == 0:
            continue

        # Auch berührende Segmente markieren (kostenlos, siehe README -
        # bei der lückenhaften OSM-Abdeckung dieser Region kaum
        # zusätzlicher Effekt, aber betrifft nie einen echten Dump mehr).
        direct_gdf = gpd.GeoDataFrame(
            {"orig_index": direct_idx},
            geometry=group.loc[direct_idx].geometry.buffer(0.1).values,
            crs=group.crs,
        )
        others = gpd.GeoDataFrame({"orig_index": group.index}, geometry=group.geometry.values, crs=group.crs)
        joined = gpd.sjoin(others, direct_gdf, predicate="intersects", lsuffix="self", rsuffix="building")
        result.loc[list(set(joined["orig_index_self"])), "is_building"] = True

    logger.info(
        "is_building: %d von %d Segmenten als Gebäude markiert (nur informativ, "
        "beeinflusst dump_pred nicht).", int(result["is_building"].sum()), len(result),
    )
    return result


def compute_is_poi(result, osm_poi_dir, buffer_m=20):
    """Fügt eine rein informative is_poi-Spalte hinzu (True/False): Segmente
    innerhalb von buffer_m Metern eines OSM-Punktes (amenity/shop/tourism -
    z.B. ein als Punkt kartiertes Restaurant, dessen Gebäude nicht als
    building=* eingezeichnet ist) oder innerhalb eines landuse=retail/
    commercial-Polygons (siehe fetch_osm_poi.py).

    Ergänzt is_building für Orte, an denen OSM nur einen Punkt/eine Fläche
    statt eines Gebäudeumrisses hat. Ändert dump_pred/dump_proba NICHT.
    Validiert an den echten Daten: betrifft 0 von 36 bekannten Dumps, fängt
    aber nur ~0.3% der grenzwertigen Falsch-Positiven ab (OSM hat in den
    Minenkacheln kaum solche Einträge) - relevanter an Ortsrändern wie
    La Negra. Für Minen ohne <mine_id>.gpkg bleibt is_poi=False."""
    result = result.copy()
    result["is_poi"] = False
    if not osm_poi_dir:
        return result

    for mine_id, group in result.groupby("mine_id"):
        poi_path = os.path.join(osm_poi_dir, f"{mine_id}.gpkg")
        if not os.path.exists(poi_path):
            continue
        poi = gpd.read_file(poi_path)
        if poi.empty:
            continue
        if poi.crs != group.crs:
            poi = poi.to_crs(group.crs)
        poi_union = poi.geometry.buffer(buffer_m).union_all()
        hit = group.geometry.intersects(poi_union).values
        result.loc[group.index[hit], "is_poi"] = True

    logger.info(
        "is_poi: %d von %d Segmenten nahe einem OSM-Punkt/Gewerbegebiet markiert "
        "(nur informativ, beeinflusst dump_pred nicht).",
        int(result["is_poi"].sum()), len(result),
    )
    return result


def compute_road_density(result, osm_roads_dir, radius_m=150, grid_threshold_m=500):
    """Fügt zwei rein informative Spalten hinzu:

    - road_density_150m: Gesamtlänge (Meter) aller OSM-Straßen (siehe
      fetch_osm_roads.py) innerhalb radius_m um den Segment-Mittelpunkt.
    - is_road_grid: road_density_150m >= grid_threshold_m.

    Ändert dump_pred/dump_proba NICHT, wie compute_is_building() - eine
    zusätzliche Spalte zum selbst Ein-/Ausblenden in QGIS.

    Idee: bloße Nähe zu IRGENDEINER Straße trennt kaum (an den echten
    Daten: 11% der 36 bekannten Dumps liegen selbst in 12m Abstand zu
    einer Straße - ein einzelner Zufahrtsweg zu einer echten Halde ist
    normal). Ein dichtes GITTER aus vielen, meist kurzen/parallelen
    Straßen ist dagegen die Signatur eines Parkplatz-/Fahrzeugbereichs
    einer Industrieanlage, nicht einer einzelnen Halde. An den echten
    Daten kalibriert (siehe README): bei grid_threshold_m=500 (Standard)
    betrifft das 0 von 36 bekannten Dumps, fängt aber 6.5% der
    grenzwertigen Falsch-Positiven (proba 0.5-0.8) - deutlich mehr als
    die einfache Gebäude-Überlappung (1.6%).

    Erwartet result mit den Spalten mine_id und geometry. Für Minen ohne
    passende <mine_id>.gpkg in osm_roads_dir bleiben beide Spalten bei
    0.0/False (kein Fehler - die OSM-Daten sind optional)."""
    result = result.copy()
    result["road_density_150m"] = 0.0
    if not osm_roads_dir:
        result["is_road_grid"] = False
        return result

    for mine_id, group in result.groupby("mine_id"):
        roads_path = os.path.join(osm_roads_dir, f"{mine_id}.gpkg")
        if not os.path.exists(roads_path):
            continue
        roads = gpd.read_file(roads_path)
        if roads.empty:
            continue
        if roads.crs != group.crs:
            roads = roads.to_crs(group.crs)

        sindex = roads.sindex
        densities = []
        for centroid in group.geometry.centroid:
            buf = centroid.buffer(radius_m)
            candidate_idx = list(sindex.intersection(buf.bounds))
            if not candidate_idx:
                densities.append(0.0)
                continue
            candidates = roads.iloc[candidate_idx]
            nearby = candidates[candidates.geometry.intersects(buf)]
            densities.append(nearby.geometry.intersection(buf).length.sum())
        result.loc[group.index, "road_density_150m"] = densities

    result["is_road_grid"] = result["road_density_150m"] >= grid_threshold_m
    logger.info(
        "road_density_150m berechnet: %d von %d Segmenten als Straßen-Gitter "
        "markiert (is_road_grid, Schwelle %.0fm, nur informativ).",
        int(result["is_road_grid"].sum()), len(result), grid_threshold_m,
    )
    return result


# ------------------------------------------------------------------
# 7) Vollprädiktion + Export für QGIS
# ------------------------------------------------------------------
def predict_and_export(feature_df, polygons_gdf, clf, feature_cols, cfg):
    X_all = feature_df[feature_cols].fillna(0.0).values
    if len(clf.classes_) < 2:
        # Das Modell wurde nur auf einer Klasse trainiert (z.B. 0 positive
        # Segmente insgesamt) -> predict_proba hätte nur 1 statt 2 Spalten
        # und würde mit IndexError abstürzen. Gleiche Situation wie in
        # train_and_evaluate(), hier für das finale Modell.
        only_class = float(clf.classes_[0])
        proba = np.full(len(X_all), only_class)
        logger.warning(
            "Finales Modell wurde nur mit Klasse %.0f trainiert (keine "
            "positiven Segmente im gesamten Datensatz) -> konstante "
            "proba=%.1f für alle Segmente.", only_class, only_class,
        )
    else:
        proba = clf.predict_proba(X_all)[:, 1]

    info_cols = [c for c in ("spatial_split_contrast", "dem_slope_mean") if c in feature_df.columns]
    result = polygons_gdf.merge(
        feature_df[["mine_id", "segment_id", "label"] + info_cols],
        on=["mine_id", "segment_id"],
    )
    threshold = cfg.get("classification_threshold", 0.5)
    result["dump_proba"] = proba
    result["dump_pred"] = (proba >= threshold).astype(int)
    require_neighbor_below = cfg.get("require_neighbor_below")
    if require_neighbor_below is not None:
        result = apply_neighbor_filter(result, high_confidence=require_neighbor_below)
    require_compact_cluster_below = cfg.get("require_compact_cluster_below")
    if require_compact_cluster_below is not None:
        result = apply_cluster_shape_filter(
            result, high_confidence=require_compact_cluster_below,
            min_compactness=cfg.get("min_cluster_compactness", 0.15),
        )

    logger.warning(
        "Die folgende Erkennungsquote nutzt das finale Modell, das auf ALLEN "
        "Minen trainiert wurde (inkl. jeder hier gezeigten). Das ist KEINE "
        "Generalisierungs-Schätzung, sondern zeigt nur, wie die exportierte "
        "Karte aussieht. Die ehrliche Zahl steht weiter oben unter "
        "'Ehrliche (Out-of-Fold-) Mine-Level-Erkennung'."
    )
    summarize_mine_detection(result)

    export_mines = cfg.get("export_mines")
    if export_mines:
        result = result[result["mine_id"].isin(export_mines)].copy()
        logger.info("Export beschränkt auf: %s (%d Segmente).", export_mines, len(result))
    osm_buildings_dir = cfg.get("osm_buildings_dir")
    if osm_buildings_dir:
        result = compute_is_building(result, osm_buildings_dir)
    osm_poi_dir = cfg.get("osm_poi_dir")
    if osm_poi_dir:
        result = compute_is_poi(result, osm_poi_dir, buffer_m=cfg.get("poi_buffer_m", 20))
    osm_roads_dir = cfg.get("osm_roads_dir")
    if osm_roads_dir:
        result = compute_road_density(
            result, osm_roads_dir,
            radius_m=cfg.get("road_density_radius_m", 150),
            grid_threshold_m=cfg.get("road_grid_threshold_m", 500),
        )

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], "segments_classified.gpkg")
    result.to_file(out_path, driver="GPKG")
    logger.info("Exportiert nach: %s", out_path)
    logger.info(
        "In QGIS laden und nach 'dump_proba' einfärben (Graduated, Schwelle "
        "~%.2f, siehe classification_threshold).", threshold,
    )
    return out_path


def report_threshold_sweep(result, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9), require_neighbor_below=None,
                            require_compact_cluster_below=None, min_cluster_compactness=0.15):
    """Loggt für mehrere Schwellwerte auf einmal, wie viele Minen mit
    bekanntem Dump erkannt werden vs. wie viele Segmente insgesamt als
    positiv vorhergesagt werden (~ Anzahl falscher Positiver, die man in
    QGIS von Hand durchsehen müsste). result["dump_proba"] sollte Out-of-
    Fold-Werte enthalten, damit der Vergleich ehrlich ist (siehe
    train_and_evaluate). Mit require_neighbor_below/require_compact_cluster_below
    werden bei jedem Schwellwert zusätzlich apply_neighbor_filter() bzw.
    apply_cluster_shape_filter() angewendet (result braucht dafür eine
    geometry-Spalte), damit sich deren Effekt direkt gegen den reinen
    Schwellwert-Effekt vergleichen lässt.

    Diese Abwägung (Erkennungsquote vs. False-Positive-Last) wurde bisher
    für jede Analyse einzeln als Wegwerf-Skript nachgerechnet - jetzt fester
    Teil der Pipeline-Ausgabe, statt bei jedem Lauf neu von Hand gebaut
    werden zu müssen."""
    labels = result["label"].values
    mine_ids = result["mine_id"].values
    dump_mine_ids = set(mine_ids[labels == 1])
    if not dump_mine_ids:
        return None

    rows = []
    for t in thresholds:
        pred_positive = result["dump_proba"].values >= t
        n_pred_pos = int(pred_positive.sum())
        detected_mine_ids = mine_ids[pred_positive]
        if require_neighbor_below is not None or require_compact_cluster_below is not None:
            tmp = result.copy()
            tmp["dump_pred"] = pred_positive.astype(int)
            if require_neighbor_below is not None:
                tmp = apply_neighbor_filter(tmp, high_confidence=require_neighbor_below)
            if require_compact_cluster_below is not None:
                tmp = apply_cluster_shape_filter(
                    tmp, high_confidence=require_compact_cluster_below,
                    min_compactness=min_cluster_compactness,
                )
            kept = tmp["dump_pred"].values == 1
            n_pred_pos = int(kept.sum())
            detected_mine_ids = mine_ids[kept]
        detected = set(detected_mine_ids) & dump_mine_ids
        rows.append({
            "threshold": t,
            "n_pred_positive": n_pred_pos,
            "mines_detected": len(detected),
            "mines_total": len(dump_mine_ids),
        })
    sweep_df = pd.DataFrame(rows)
    logger.info(
        "Schwellwert-Vergleich (Out-of-Fold, ehrlich): niedrigerer Schwellwert "
        "= mehr erkannte Minen, aber auch mehr falsch-positive Segmente zum "
        "manuellen Durchsehen in QGIS:\n%s",
        sweep_df.to_string(index=False),
    )
    return sweep_df


def apply_neighbor_filter(result, high_confidence=0.6):
    """Setzt dump_pred für Segmente mit dump_proba < high_confidence auf 0
    zurück, wenn kein räumlich angrenzendes Segment in derselben Mine
    ebenfalls positiv vorhergesagt wurde. Segmente mit dump_proba >=
    high_confidence bleiben unangetastet, auch ohne Nachbarn.

    Grund für die Konfidenz-Schwelle statt eines pauschalen Cluster-Filters:
    manche echten Dumps bestehen aus nur einem Segment (z.B. mine_012 - der
    einzige bekannte Dump dort ist ein einzelnes, hochkonfident erkanntes
    Segment). Ein pauschaler "mindestens 2 Segmente"-Filter hätte genau
    diesen Treffer gelöscht. Isolierte EINZELNE Segmente mit nur mittlerer
    Konfidenz entlang von Fahrstraßen/Gruben-Rändern sind dagegen ein
    wiederkehrendes Muster bei den Falsch-Positiven (siehe Analyse) - für
    die gilt der Filter.

    Erwartet result mit den Spalten mine_id, dump_proba, dump_pred und
    geometry (Polygone)."""
    result = result.copy()
    result["dump_pred"] = result["dump_pred"].astype(int)
    positive = result[result["dump_pred"] == 1]
    if positive.empty:
        return result

    to_downgrade = []
    for mine_id, group in positive.groupby("mine_id"):
        low_conf = group[group["dump_proba"] < high_confidence]
        if low_conf.empty:
            continue
        # Kleiner Puffer gegen Fließkomma-Ungenauigkeiten an Segmentgrenzen
        # (z.B. nach einer Reprojektion), damit tatsächlich angrenzende
        # Segmente nicht knapp als "nicht berührend" durchrutschen.
        low_gdf = gpd.GeoDataFrame(
            {"orig_index": low_conf.index}, geometry=low_conf.geometry.buffer(0.1).values,
            crs=group.crs,
        )
        others = gpd.GeoDataFrame(
            {"orig_index": group.index}, geometry=group.geometry.values, crs=group.crs,
        )
        joined = gpd.sjoin(low_gdf, others, predicate="intersects", lsuffix="low", rsuffix="other")
        supported = set(joined.loc[
            joined["orig_index_low"] != joined["orig_index_other"], "orig_index_low"
        ])
        to_downgrade.extend(set(low_conf.index) - supported)

    if to_downgrade:
        result.loc[to_downgrade, "dump_pred"] = 0
    return result


def apply_cluster_shape_filter(result, high_confidence=0.6, min_compactness=0.15):
    """Setzt dump_pred für Segmente mit dump_proba < high_confidence auf 0
    zurück, wenn das zusammenhängende Cluster berührender positiver
    Segmente in derselben Mine, zu dem sie gehören, eine Kompaktheit
    (4π·Fläche/Umfang², wie shape_compactness) unter min_compactness hat.
    Segmente mit dump_proba >= high_confidence bleiben unangetastet,
    gleiche Begründung wie bei apply_neighbor_filter().

    Ergänzt apply_neighbor_filter(), statt es zu ersetzen: eine lange Kette
    berührender Segmente entlang einer Straße oder Klippenkante besteht
    zwar aus lauter gegenseitig unterstützten Nachbarn (würde den
    Nachbarschafts-Filter also unbeschadet passieren), ist aber lang und
    dünn statt haufenförmig wie ein echter Dump - genau das misst die
    Cluster-Kompaktheit. Robust gegenüber gekrümmten/kurvigen Ketten,
    anders als ein Seitenverhältnis-Test der Bounding Box (der bei
    kurvigen Straßen versagt - eine gewundene Straße kann trotzdem in eine
    eher quadratische Bounding Box passen).

    An den echten Daten kalibriert: mehrsegmentige Falsch-Positiv-Cluster
    haben median Kompaktheit 0.27 (20% liegen unter 0.15), mehrsegmentige
    echte Dump-Cluster median 0.34 (0% liegen unter 0.15) - der
    Standardwert 0.15 verwirft damit einen Teil der dünnen Ketten, ohne in
    den vorliegenden Daten einen einzigen echten mehrsegmentigen Dump zu
    verlieren.

    Erwartet result mit den Spalten mine_id, dump_proba, dump_pred und
    geometry (Polygone)."""
    result = result.copy()
    result["dump_pred"] = result["dump_pred"].astype(int)
    positive = result[result["dump_pred"] == 1]
    if positive.empty:
        return result

    to_downgrade = []
    for mine_id, group in positive.groupby("mine_id"):
        merged = unary_union(group.geometry.tolist())
        parts = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
        for part in parts:
            if part.area <= 0:
                continue
            compactness = 4 * np.pi * part.area / (part.length ** 2 + 1e-9)
            if compactness >= min_compactness:
                continue
            members = group[group.geometry.intersects(part.buffer(0.1))]
            low_conf_members = members[members["dump_proba"] < high_confidence]
            to_downgrade.extend(low_conf_members.index)

    if to_downgrade:
        result.loc[to_downgrade, "dump_pred"] = 0
    return result


def summarize_mine_detection(result):
    """Fasst pro Mine zusammen, ob ein bekannter Dump überhaupt gefunden
    wurde (mindestens ein Segment korrekt als positiv vorhergesagt).
    Bei nur einer Handvoll positiver Minen ist das aussagekräftiger als
    ein einzelner Gesamt-Score über alle Segmente hinweg - eine Mine mit
    vielen Segmenten kann den Gesamt-F1 dominieren, ohne dass klar wird,
    ob die tatsächlich interessanten (seltenen) Dump-Minen erkannt wurden."""
    by_mine = result.groupby("mine_id").agg(
        n_segments=("segment_id", "count"),
        n_true_positive=("label", "sum"),
        n_pred_positive=("dump_pred", "sum"),
    ).reset_index()

    dump_mines = by_mine[by_mine["n_true_positive"] > 0].copy()
    if dump_mines.empty:
        logger.info(
            "Keine Mine mit bestätigtem Dump im Ergebnis - keine "
            "Mine-Level-Erkennungsübersicht möglich."
        )
        return dump_mines

    dump_mines["detected"] = dump_mines["n_pred_positive"] > 0
    n_detected = int(dump_mines["detected"].sum())
    n_total = len(dump_mines)

    logger.info(
        "Mine-Level-Erkennung: bei wie vielen der %d Minen mit bekanntem Dump "
        "wurde mindestens ein Segment korrekt als positiv erkannt?", n_total,
    )
    for _, row in dump_mines.sort_values("mine_id").iterrows():
        flag = "GEFUNDEN" if row["detected"] else "NICHT gefunden"
        logger.info(
            "  %s: %d echte positive Segmente, %d vorhergesagt positiv  [%s]",
            row["mine_id"], row["n_true_positive"], row["n_pred_positive"], flag,
        )
    logger.info(
        "Erkannt: %d von %d bekannten Dump-Minen (%.0f%%)",
        n_detected, n_total, 100 * n_detected / n_total,
    )
    return dump_mines


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def apply_extra_labels(feature_df, polygons_gdf, labels_gdf, min_overlap_ratio):
    """Setzt zusätzliche Halden-Labels (z.B. in QGIS geprüfte Funde) auf einen
    FERTIGEN Segment-Datensatz an, ohne ihn neu zu berechnen: Segmente, die eines
    der Polygone zu >= min_overlap_ratio überdecken, werden positiv. Bestehende
    Positive bleiben; die Zuordnung läuft über mine_id des Label-Polygons."""
    labels_gdf = labels_gdf.to_crs(polygons_gdf.crs)
    feature_df = feature_df.copy()
    mine_ids = feature_df["mine_id"].astype(str)
    added = 0
    for mine, lab in labels_gdf.groupby(labels_gdf["mine_id"].astype(str)):
        polys = polygons_gdf[polygons_gdf["mine_id"].astype(str) == mine]
        if polys.empty:
            logger.warning("Zusatz-Labels für '%s': keine Segmente dieser Mine/Kachel im Datensatz - übersprungen.", mine)
            continue
        labeled = label_segments(polys, lab, min_overlap_ratio)
        positive_ids = set(labeled.loc[labeled["label"] == 1, "segment_id"])
        rows = feature_df.index[(mine_ids == mine).values]
        newly = feature_df.loc[rows, "segment_id"].isin(positive_ids).values & (feature_df.loc[rows, "label"].values == 0)
        feature_df.loc[rows[newly], "label"] = 1
        added += int(newly.sum())
    logger.info("Zusatz-Labels: %d Segmente zusätzlich positiv (%d Label-Polygone).", added, len(labels_gdf))
    return feature_df


def mark_hard_negatives(feature_df, polygons_gdf, negatives_gdf, min_overlap=0.5):
    """Bool-Array (Länge feature_df): label=0-Segmente, die zu >= min_overlap von einem
    bestätigten Fehlalarm-Polygon (negatives_gdf, Spalte mine_id) überdeckt werden."""
    negatives_gdf = negatives_gdf.to_crs(polygons_gdf.crs)
    mine_ids = feature_df["mine_id"].astype(str).values
    mask = np.zeros(len(feature_df), dtype=bool)
    for mine, neg in negatives_gdf.groupby(negatives_gdf["mine_id"].astype(str)):
        polys = polygons_gdf[polygons_gdf["mine_id"].astype(str) == mine]
        if polys.empty:
            continue
        labeled = label_segments(polys, neg, min_overlap)
        ids = set(labeled.loc[labeled["label"] == 1, "segment_id"])
        mask |= (mine_ids == mine) & feature_df["segment_id"].isin(ids).values & (feature_df["label"].values == 0)
    logger.info("Harte Negative: %d label=0-Segmente in bestätigten Fehlalarm-Flächen.", int(mask.sum()))
    return mask


def restrict_to_reviewed(feature_df, polygons_gdf, hard_mask, tile_ids):
    """Für Kacheln, in denen nur einzelne Stellen geprüft wurden (z.B. Steinbrüche): im
    TRAININGS-Datensatz bleiben von diesen Kacheln nur positive Segmente und label=0-Segmente
    in bestätigten Fehlalarm-Flächen (hard_mask). Alle übrigen label=0-Segmente dort sind
    UNGEPRÜFT (dort kann eine unbekannte Halde liegen) und dürfen nicht als 'kein Dump' lernen.
    Gibt (feature_df, polygons_gdf, hard_mask) neu indiziert zurück."""
    tile_ids = set(map(str, tile_ids))
    in_tile = feature_df["mine_id"].astype(str).isin(tile_ids).values
    keep = ~(in_tile & (feature_df["label"].values == 0) & ~np.asarray(hard_mask, dtype=bool))
    dropped = feature_df.loc[~keep, ["mine_id", "segment_id"]]
    drop_keys = set(zip(dropped["mine_id"], dropped["segment_id"]))
    keep_poly = np.array([(m, sid) not in drop_keys
                          for m, sid in zip(polygons_gdf["mine_id"], polygons_gdf["segment_id"])])
    logger.info("Nur-geprüfte-Stellen-Kacheln: %d ungeprüfte label=0-Segmente aus dem Training genommen.",
                int((~keep).sum()))
    return (feature_df[keep].reset_index(drop=True), polygons_gdf[keep_poly].reset_index(drop=True),
            np.asarray(hard_mask, dtype=bool)[keep])


def drop_ignored_negatives(feature_df, polygons_gdf, points_path, radius_m):
    """Entfernt aus dem TRAININGS-Datensatz alle label=0-Segmente, die innerhalb
    radius_m um einen Punkt aus points_path liegen (positive Segmente bleiben).

    Zweck: an bekannten Halden-Orten, die nicht (oder nur teilweise) als
    Label-Polygon eingezeichnet sind, würden die umliegenden Reifen-Segmente
    sonst als "sicher kein Dump" mittrainiert. Die Segmente werden nur aus
    dem Training genommen, nicht aus Vorhersage und Export."""
    # mehrere Dateien mit Komma getrennt möglich (z.B. Boss-Punkte + geprüfte Standorte)
    pts = pd.concat([gpd.read_file(p.strip()).to_crs(polygons_gdf.crs)[["geometry"]]
                     for p in str(points_path).split(",") if p.strip()], ignore_index=True)
    zone = pts.geometry.buffer(radius_m).union_all()
    in_zone = polygons_gdf[polygons_gdf.geometry.intersects(zone)][["mine_id", "segment_id"]]
    keys = set(zip(in_zone["mine_id"], in_zone["segment_id"]))
    flagged = np.array([(m, sid) in keys for m, sid in zip(feature_df["mine_id"], feature_df["segment_id"])])
    drop = flagged & (feature_df["label"].values == 0)
    kept = feature_df[~drop].reset_index(drop=True)
    dropped_keys = set(zip(feature_df.loc[drop, "mine_id"], feature_df.loc[drop, "segment_id"]))
    keep_poly = np.array([(m, sid) not in dropped_keys
                          for m, sid in zip(polygons_gdf["mine_id"], polygons_gdf["segment_id"])])
    logger.info("Ignorier-Zone (%.0f m um %d Punkte): %d label=0-Segmente aus dem Training genommen.",
                radius_m, len(pts), int(drop.sum()))
    return kept, polygons_gdf[keep_poly].reset_index(drop=True)


def main(cfg=CONFIG):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cache_features = os.path.join(cfg["output_dir"], "feature_cache.pkl")
    cache_polygons = os.path.join(cfg["output_dir"], "polygons_cache.gpkg")

    feature_df = None
    if cfg.get("use_cache") and os.path.exists(cache_features) and os.path.exists(cache_polygons):
        # Das Segmentieren + Merkmale-Berechnen ist der mit Abstand teuerste
        # Schritt (bei den echten Daten ca. 25 Minuten). Für schnelles
        # Iterieren an Schwellwerten/Modell-Einstellungen lohnt es sich,
        # diesen Schritt nicht bei jedem Versuch zu wiederholen.
        logger.info("1) Lade zwischengespeicherten Segment-Datensatz aus %s ...", cfg["output_dir"])
        feature_df = pd.read_pickle(cache_features)
        polygons_gdf = gpd.read_file(cache_polygons)
        for key, names, label in (("s1_dir", S1_BAND_NAMES, "Sentinel-1"),
                                  ("dem_dir", DEM_BAND_NAMES, "Gelände"),
                                  ("s2t_dir", S2T_BAND_NAMES, "Zeitreihen"),
                                  ("s2my_dir", select_s2my_names(cfg) if cfg.get("s2my_dir") else S2MY_BAND_NAMES, "Mehrjahres"),
                                  ("s1t_dir", S1T_BAND_NAMES, "Radar-Zeitreihen")):
            if cfg.get(key) and f"{names[0]}_mean" not in feature_df.columns:
                logger.warning(
                    "Zwischenspeicher enthält keine %s-Merkmale, %s ist aber "
                    "gesetzt -> Segment-Datensatz wird neu berechnet.", label, key,
                )
                feature_df = None
                break

    extra_dir = cfg.get("extra_imagery_dir")
    labeled_dir = cfg.get("labeled_imagery_dir")
    reviewed_dir = cfg.get("reviewed_imagery_dir")
    extra_ids, labeled_ids, reviewed_ids = [], [], []
    if extra_dir or labeled_dir or reviewed_dir:
        if feature_df is None:
            raise ValueError(
                "extra_imagery_dir/labeled_imagery_dir brauchen einen vorhandenen Zwischenspeicher: "
                "mit --use-cache und einem output_dir starten, das feature_cache.pkl "
                "und polygons_cache.gpkg eines früheren Volllaufs enthält."
            )
        new_feats, new_polys = [], []
        if labeled_dir:
            if not cfg.get("extra_labels_path"):
                raise ValueError("labeled_imagery_dir braucht extra_labels_path (Label-Polygone mit mine_id = Kachelname).")
            logger.info("1b) Segmentiere gelabelte Kacheln aus %s (werden mittrainiert) ...", labeled_dir)
            lf, lp = build_dataset({**cfg, "imagery_dir": labeled_dir, "labels_path": cfg["extra_labels_path"]})
            labeled_ids = sorted(lf["mine_id"].unique())
            new_feats.append(lf); new_polys.append(lp)
        if extra_dir:
            logger.info("1c) Segmentiere zusätzliche Kacheln aus %s (werden nur bewertet) ...", extra_dir)
            ef, ep = build_dataset({**cfg, "imagery_dir": extra_dir})
            extra_ids = sorted(ef["mine_id"].unique())
            new_feats.append(ef); new_polys.append(ep)
        if reviewed_dir:
            logger.info("1d) Segmentiere Kacheln mit geprüften Stellen aus %s (trainiert wird nur auf den "
                        "geprüften Stellen, bewertet wird alles) ...", reviewed_dir)
            rf, rp = build_dataset({**cfg, "imagery_dir": reviewed_dir})
            reviewed_ids = sorted(rf["mine_id"].unique())
            new_feats.append(rf); new_polys.append(rp)
        replaced = extra_ids + labeled_ids + reviewed_ids
        # Die Segment-Polygone der neuen Kacheln stehen NICHT im polygons_cache.gpkg
        # (der gehört zum Volllauf) - separat ablegen, damit sich z.B.
        # eval_segment_size.py für diese Kacheln auswerten lässt.
        gpd.GeoDataFrame(pd.concat(new_polys, ignore_index=True), crs=polygons_gdf.crs).to_file(
            os.path.join(cfg["output_dir"], "polygons_extra.gpkg"), driver="GPKG"
        )
        feature_df = pd.concat([feature_df[~feature_df["mine_id"].isin(replaced)]] + new_feats, ignore_index=True)
        polygons_gdf = gpd.GeoDataFrame(
            pd.concat([polygons_gdf[~polygons_gdf["mine_id"].isin(replaced)]] + new_polys, ignore_index=True),
            crs=polygons_gdf.crs,
        )
        cfg = {**cfg, "export_mines": (extra_ids + reviewed_ids) or labeled_ids}

    if feature_df is None:
        logger.info("1) Baue Segment-Datensatz aus allen Minen auf ...")
        feature_df, polygons_gdf = build_dataset(cfg)
        os.makedirs(cfg["output_dir"], exist_ok=True)
        feature_df.to_pickle(cache_features)
        polygons_gdf.to_file(cache_polygons, driver="GPKG")
        logger.info(
            "Segment-Datensatz zwischengespeichert in %s (mit --use-cache beim "
            "nächsten Mal wiederverwenden, ohne alles neu zu berechnen).",
            cfg["output_dir"],
        )

    if cfg.get("add_labels_path"):
        feature_df = apply_extra_labels(
            feature_df, polygons_gdf, gpd.read_file(cfg["add_labels_path"]), cfg["min_overlap_ratio"])

    logger.info(
        "Gesamt: %d Segmente, %d positiv (%d Minen).",
        len(feature_df), int(feature_df["label"].sum()), feature_df["mine_id"].nunique(),
    )

    logger.info("2) Training + räumliche Kreuzvalidierung (GroupKFold nach Mine) ...")
    # Neue, ungelabelte Kacheln (extra_imagery_dir) werden nur bewertet, nie zum
    # Training verwendet - sonst würden ihre label=0-Segmente als bekannte
    # Nicht-Dumps mittrainiert.
    trained_on = ~feature_df["mine_id"].isin(extra_ids)
    poly_trained_on = ~polygons_gdf["mine_id"].isin(extra_ids)
    train_feat = feature_df[trained_on].reset_index(drop=True)
    train_poly = polygons_gdf[poly_trained_on].reset_index(drop=True)
    if cfg.get("ignore_points_path"):
        train_feat, train_poly = drop_ignored_negatives(
            train_feat, train_poly, cfg["ignore_points_path"], cfg.get("ignore_radius_m", 150.0),
        )
    hard_mask = None
    if cfg.get("hard_negatives_path"):
        hard_mask = mark_hard_negatives(train_feat, train_poly, gpd.read_file(cfg["hard_negatives_path"]))
        logger.info("Harte Negative werden im Training %dx gewichtet.", int(cfg.get("hard_negative_factor", 10)))
    if reviewed_ids:
        if hard_mask is None:
            hard_mask = np.zeros(len(train_feat), dtype=bool)
        train_feat, train_poly, hard_mask = restrict_to_reviewed(train_feat, train_poly, hard_mask, reviewed_ids)
        if not hard_mask.any():
            hard_mask = None
    if hard_mask is None:
        clf, feature_cols, scores_df = train_and_evaluate(train_feat, train_poly, cfg)
    else:
        clf, feature_cols, scores_df = train_and_evaluate(
            train_feat, train_poly, cfg, hard_neg_mask=hard_mask,
            hard_neg_factor=int(cfg.get("hard_negative_factor", 10)))

    logger.info("3) Vollprädiktion über alle Segmente + Export als GeoPackage ...")
    predict_and_export(feature_df, polygons_gdf, clf, feature_cols, cfg)


if __name__ == "__main__":
    from cli import build_arg_parser, resolve_config

    parser = build_arg_parser(
        "OBIA-Pipeline: klassifiziert Sentinel-2-Segmente als Reifenhalde oder nicht."
    )
    cli_args = parser.parse_args()
    main(resolve_config(CONFIG, cli_args))
