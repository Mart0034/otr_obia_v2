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

# ------------------------------------------------------------------
# KONFIGURATION - hier an die eigenen Pfade/Daten anpassen
# ------------------------------------------------------------------
CONFIG = {
    "imagery_dir": "data/imagery",              # ein GeoTIFF pro Mine
    "labels_path": "data/dump_labels.gpkg",      # digitalisierte Dump-Polygone
    "mine_id_field": "mine_id",                  # Spaltenname in labels_path
    "n_segments_per_mine": 800,                  # SLIC-Zielanzahl Superpixel
    "compactness": 8,                             # SLIC: Form- vs. Farbtreue
    "min_overlap_ratio": 0.30,                    # Segment = positiv, wenn
                                                   # >= 30% seiner Fläche im
                                                   # gelabelten Dump liegt
    "band_names": [
        "blue", "green", "red", "nir", "swir1", "swir2",
        "ndvi", "ndwi", "dsi", "bsi",
    ],
    "texture_band": "dsi",                        # Band für GLCM-Textur
    "n_estimators": 400,
    "random_state": 42,
    "output_dir": "output",
}


# ------------------------------------------------------------------
# 1) Rasterdaten laden
# ------------------------------------------------------------------
def load_mine_raster(path):
    """Lädt einen Sentinel-2-Stack (Bänder, Höhe, Breite) + Georeferenz."""
    with rasterio.open(path) as src:
        arr = src.read()  # shape: (bands, H, W)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    arr = np.moveaxis(arr, 0, -1)  # -> (H, W, bands)
    return arr, transform, crs, nodata


def extract_mine_id(filename, pattern=r"(mine[_\-]?\d+|\d+)"):
    """Versucht, die mine_id aus dem Dateinamen zu extrahieren.
    Bei Bedarf an das tatsächliche Namensschema anpassen."""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(pattern, base, flags=re.IGNORECASE)
    return m.group(0) if m else base


# ------------------------------------------------------------------
# 2) Segmentierung
# ------------------------------------------------------------------
def segment_mine(arr, n_segments, compactness):
    """SLIC-Superpixel-Segmentierung auf allen Bändern gleichzeitig.
    Ersetzt die GRASS-i.segment-Variante aus QGIS/GRASS, funktioniert
    aber identisch im Prinzip (Region-Growing/Clustering) und ist
    reiner Python-Code ohne GRASS-Abhängigkeit."""
    # NaNs/Inf robust behandeln
    safe_arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # SLIC erwartet vernünftig skalierte Werte -> pro Band normieren
    norm = np.zeros_like(safe_arr, dtype=np.float32)
    for b in range(safe_arr.shape[-1]):
        band = safe_arr[..., b]
        lo, hi = np.percentile(band, [2, 98])
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
    return gdf


# ------------------------------------------------------------------
# 3) Merkmale pro Segment
# ------------------------------------------------------------------
def compute_segment_features(arr, segments, band_names, texture_band):
    """Berechnet je Segment: Mittelwert & Std pro Band + GLCM-Textur
    auf dem gewählten Band (z. B. Dark Surface Index)."""
    seg_ids = np.unique(segments)
    seg_ids = seg_ids[seg_ids > 0]

    tex_idx = band_names.index(texture_band)
    tex_band = arr[..., tex_idx]
    # für GLCM auf 8-Bit-Stufen quantisieren
    tex_band_q = np.clip(
        ((tex_band - np.nanmin(tex_band)) /
         (np.nanmax(tex_band) - np.nanmin(tex_band) + 1e-6) * 255),
        0, 255,
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
    seg_gdf["area"] = seg_gdf.geometry.area

    if labels_gdf.empty:
        seg_gdf["label"] = 0
        return seg_gdf

    overlay = gpd.overlay(seg_gdf, labels_gdf[["geometry"]], how="intersection")
    overlay["overlap_area"] = overlay.geometry.area
    overlap_by_seg = overlay.groupby(["mine_id", "segment_id"])["overlap_area"].sum()

    seg_gdf = seg_gdf.set_index(["mine_id", "segment_id"])
    seg_gdf["overlap_area"] = overlap_by_seg.reindex(seg_gdf.index).fillna(0.0)
    seg_gdf["overlap_ratio"] = seg_gdf["overlap_area"] / seg_gdf["area"]
    seg_gdf["label"] = (seg_gdf["overlap_ratio"] >= min_overlap_ratio).astype(int)
    seg_gdf = seg_gdf.reset_index()
    return seg_gdf


# ------------------------------------------------------------------
# 5) Gesamten Datensatz aus allen Minen aufbauen
# ------------------------------------------------------------------
def build_dataset(cfg):
    labels_gdf = gpd.read_file(cfg["labels_path"])
    imagery_paths = sorted(glob.glob(os.path.join(cfg["imagery_dir"], "*.tif")))
    if not imagery_paths:
        raise FileNotFoundError(f"Keine GeoTIFFs in {cfg['imagery_dir']} gefunden.")

    all_features = []
    all_polygons = []

    for path in imagery_paths:
        mine_id = extract_mine_id(path)
        arr, transform, crs, nodata = load_mine_raster(path)

        segments = segment_mine(arr, cfg["n_segments_per_mine"], cfg["compactness"])
        seg_polys = segments_to_polygons(segments, transform, crs, mine_id)

        mine_labels = labels_gdf[labels_gdf[cfg["mine_id_field"]] == mine_id]
        if not mine_labels.empty and mine_labels.crs != crs:
            mine_labels = mine_labels.to_crs(crs)

        seg_polys = label_segments(seg_polys, mine_labels, cfg["min_overlap_ratio"])

        feats = compute_segment_features(arr, segments, cfg["band_names"], cfg["texture_band"])
        merged = seg_polys.merge(feats, on="segment_id", how="inner")

        all_features.append(pd.DataFrame(merged.drop(columns="geometry")))
        all_polygons.append(merged[["mine_id", "segment_id", "geometry"]])

        n_pos = int(merged["label"].sum())
        print(f"  {mine_id}: {len(merged)} Segmente, davon {n_pos} positiv")

    feature_df = pd.concat(all_features, ignore_index=True)
    polygons_gdf = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True), crs=crs)
    return feature_df, polygons_gdf


# ------------------------------------------------------------------
# 6) Training + räumliche Validierung (Split NACH Mine, wie im Report)
# ------------------------------------------------------------------
def train_and_evaluate(feature_df, cfg):
    feature_cols = [c for c in feature_df.columns
                     if c not in ("segment_id", "mine_id", "label", "area",
                                  "overlap_area", "overlap_ratio")]
    X = feature_df[feature_cols].fillna(0.0).values
    y = feature_df["label"].values
    groups = feature_df["mine_id"].values

    n_splits = min(5, feature_df["mine_id"].nunique())
    gkf = GroupKFold(n_splits=n_splits)

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        clf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            class_weight="balanced",
            random_state=cfg["random_state"],
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[val_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)

        f1 = f1_score(y[val_idx], pred, zero_division=0)
        try:
            auc = roc_auc_score(y[val_idx], proba)
        except ValueError:
            auc = float("nan")  # falls eine Fold nur eine Klasse enthält

        fold_scores.append({"fold": fold, "f1": f1, "auc": auc})
        print(f"  Fold {fold}: F1={f1:.3f}  AUC={auc:.3f}")

    scores_df = pd.DataFrame(fold_scores)
    print("\nMittelwerte über alle Folds:")
    print(scores_df[["f1", "auc"]].mean())

    # finales Modell auf ALLEN Daten für die spätere Vollprädiktion
    final_clf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        class_weight="balanced",
        random_state=cfg["random_state"],
        n_jobs=-1,
    )
    final_clf.fit(X, y)

    importances = pd.Series(final_clf.feature_importances_, index=feature_cols)
    print("\nWichtigste Merkmale:")
    print(importances.sort_values(ascending=False).head(10))

    return final_clf, feature_cols, scores_df


# ------------------------------------------------------------------
# 7) Vollprädiktion + Export für QGIS
# ------------------------------------------------------------------
def predict_and_export(feature_df, polygons_gdf, clf, feature_cols, cfg):
    X_all = feature_df[feature_cols].fillna(0.0).values
    proba = clf.predict_proba(X_all)[:, 1]

    result = polygons_gdf.merge(
        feature_df[["mine_id", "segment_id", "label"]],
        on=["mine_id", "segment_id"],
    )
    result["dump_proba"] = proba
    result["dump_pred"] = (proba >= 0.5).astype(int)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], "segments_classified.gpkg")
    result.to_file(out_path, driver="GPKG")
    print(f"\nExportiert nach: {out_path}")
    print("In QGIS laden und nach 'dump_proba' einfärben (Graduated, Schwelle ~0.5).")
    return out_path


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main(cfg=CONFIG):
    print("1) Baue Segment-Datensatz aus allen Minen auf ...")
    feature_df, polygons_gdf = build_dataset(cfg)

    print(f"\nGesamt: {len(feature_df)} Segmente, "
          f"{int(feature_df['label'].sum())} positiv "
          f"({feature_df['mine_id'].nunique()} Minen).")

    print("\n2) Training + räumliche Kreuzvalidierung (GroupKFold nach Mine) ...")
    clf, feature_cols, scores_df = train_and_evaluate(feature_df, cfg)

    print("\n3) Vollprädiktion über alle Segmente + Export als GeoPackage ...")
    predict_and_export(feature_df, polygons_gdf, clf, feature_cols, cfg)


if __name__ == "__main__":
    main()
