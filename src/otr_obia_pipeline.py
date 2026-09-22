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
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shapely_shape
from skimage.segmentation import slic
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, f1_score, roc_auc_score

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
    "output_dir": "output",
}


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
    mine_id = m.group(0) if m else base
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

    segments = slic(
        norm,
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=-1,
        start_label=1,
        enforce_connectivity=True,
        mask=valid_mask,
    )
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
    """Berechnet geometrische Formmerkmale je Segment: Fläche, Umfang und
    Kompaktheit (Polsby-Popper-Maß: 1.0 = perfekter Kreis, kleiner = länglich
    oder verwinkelt). Wird in der Modul-Doku als "Formmerkmale" angekündigt,
    fehlte aber bisher in der Implementierung."""
    seg_gdf = seg_gdf.copy()
    seg_gdf["shape_area"] = seg_gdf.geometry.area
    seg_gdf["shape_perimeter"] = seg_gdf.geometry.length
    seg_gdf["shape_compactness"] = (
        4 * np.pi * seg_gdf["shape_area"] / (seg_gdf["shape_perimeter"] ** 2 + 1e-9)
    )
    return seg_gdf


def compute_segment_features(arr, segments, band_names, texture_band):
    """Berechnet je Segment: Mittelwert & Std pro Band + GLCM-Textur
    auf dem gewählten Band (z. B. Dark Surface Index)."""
    seg_ids = np.unique(segments)
    seg_ids = seg_ids[seg_ids > 0]

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

    rows = []
    for seg_id in seg_ids:
        m = segments == seg_id
        if m.sum() < 4:
            continue  # zu kleines Segment, überspringen

        row = {"segment_id": int(seg_id), "n_pixels": int(m.sum())}

        for b, name in enumerate(band_names):
            vals = arr[..., b][m]
            row[f"{name}_mean"] = float(np.nanmean(vals))
            row[f"{name}_std"] = float(np.nanstd(vals))

        # GLCM-Textur auf der Bounding Box des Segments (einfache,
        # robuste Näherung statt exakter Segmentgeometrie)
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        patch = tex_band_q[y0:y1, x0:x1]
        if patch.shape[0] >= 2 and patch.shape[1] >= 2:
            glcm = graycomatrix(
                patch, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256, symmetric=True, normed=True,
            )
            row["tex_contrast"] = float(graycoprops(glcm, "contrast").mean())
            row["tex_homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
            row["tex_energy"] = float(graycoprops(glcm, "energy").mean())
            row["tex_correlation"] = float(np.nan_to_num(graycoprops(glcm, "correlation")).mean())
        else:
            row["tex_contrast"] = row["tex_homogeneity"] = 0.0
            row["tex_energy"] = row["tex_correlation"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


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


def _process_one_mine(path, cfg, labels_gdf, target_crs):
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

    feats = compute_segment_features(arr, segments, cfg["band_names"], cfg["texture_band"])
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

    imagery_paths = sorted(glob.glob(os.path.join(cfg["imagery_dir"], "*.tif")))
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

    results = [None] * len(imagery_paths)
    if n_jobs <= 1:
        for i, path in enumerate(imagery_paths):
            results[i] = _process_one_mine(path, cfg, labels_gdf, target_crs)
    else:
        logger.info(
            "Verarbeite %d Minen parallel mit %d Prozessen ...",
            len(imagery_paths), n_jobs,
        )
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker_logging) as executor:
            future_to_idx = {
                executor.submit(_process_one_mine, path, cfg, labels_gdf, target_crs): i
                for i, path in enumerate(imagery_paths)
            }
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()

    all_features = [feats for feats, _ in results]
    all_polygons = [polys for _, polys in results]

    feature_df = pd.concat(all_features, ignore_index=True)
    polygons_gdf = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True), crs=target_crs)
    return feature_df, polygons_gdf


# ------------------------------------------------------------------
# 6) Training + räumliche Validierung (Split NACH Mine, wie im Report)
# ------------------------------------------------------------------
def train_and_evaluate(feature_df, cfg):
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

    n_mines = feature_df["mine_id"].nunique()
    n_splits = min(5, n_mines)
    if n_splits < 5:
        logger.warning(
            "Nur %d Minen im Datensatz -> GroupKFold läuft mit n_splits=%d statt 5.",
            n_mines, n_splits,
        )
    gkf = GroupKFold(n_splits=n_splits)

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        val_mines = sorted(set(groups[val_idx]))
        clf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            class_weight="balanced",
            random_state=cfg["random_state"],
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        if len(clf.classes_) < 2:
            # Trainings-Split dieser Fold enthält nur eine Klasse (z.B. eine
            # Mine ganz ohne positive Segmente) -> predict_proba hätte nur
            # 1 statt 2 Spalten und würde mit IndexError abstürzen.
            only_class = float(clf.classes_[0])
            proba = np.full(len(val_idx), only_class)
            logger.warning(
                "Fold %d (Validierungs-Minen: %s): Trainingsdaten enthalten nur "
                "Klasse %.0f -> Modell kann nicht lernen, verwende konstante "
                "proba=%.1f für diese Fold.", fold, val_mines, only_class, only_class,
            )
        else:
            proba = clf.predict_proba(X[val_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)

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

    # finales Modell auf ALLEN Daten für die spätere Vollprädiktion
    final_clf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        class_weight="balanced",
        random_state=cfg["random_state"],
        n_jobs=-1,
    )
    final_clf.fit(X, y)

    importances = pd.Series(final_clf.feature_importances_, index=feature_cols)
    logger.info(
        "Wichtigste Merkmale:\n%s", importances.sort_values(ascending=False).head(10)
    )

    return final_clf, feature_cols, scores_df


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

    result = polygons_gdf.merge(
        feature_df[["mine_id", "segment_id", "label"]],
        on=["mine_id", "segment_id"],
    )
    result["dump_proba"] = proba
    result["dump_pred"] = (proba >= 0.5).astype(int)

    summarize_mine_detection(result)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], "segments_classified.gpkg")
    result.to_file(out_path, driver="GPKG")
    logger.info("Exportiert nach: %s", out_path)
    logger.info("In QGIS laden und nach 'dump_proba' einfärben (Graduated, Schwelle ~0.5).")
    return out_path


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
def main(cfg=CONFIG):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("1) Baue Segment-Datensatz aus allen Minen auf ...")
    feature_df, polygons_gdf = build_dataset(cfg)

    logger.info(
        "Gesamt: %d Segmente, %d positiv (%d Minen).",
        len(feature_df), int(feature_df["label"].sum()), feature_df["mine_id"].nunique(),
    )

    logger.info("2) Training + räumliche Kreuzvalidierung (GroupKFold nach Mine) ...")
    clf, feature_cols, scores_df = train_and_evaluate(feature_df, cfg)

    logger.info("3) Vollprädiktion über alle Segmente + Export als GeoPackage ...")
    predict_and_export(feature_df, polygons_gdf, clf, feature_cols, cfg)


if __name__ == "__main__":
    from cli import build_arg_parser, resolve_config

    parser = build_arg_parser(
        "OBIA-Pipeline: klassifiziert Sentinel-2-Segmente als Reifenhalde oder nicht."
    )
    cli_args = parser.parse_args()
    main(resolve_config(CONFIG, cli_args))
