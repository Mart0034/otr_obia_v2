"""Gemeinsame Kommandozeilen-/Konfigurations-Hilfsfunktionen, die sowohl
von der Pipeline (otr_obia_pipeline.py) als auch vom Daten-Check-Skript
(check_data.py) verwendet werden, damit man Einstellungen nicht mehr im
Code selbst ändern muss."""
import argparse
import json


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "Pfad zu einer JSON-Datei mit Einstellungen, die die "
            'Standardwerte aus CONFIG überschreiben, z.B. '
            '{"imagery_dir": "...", "min_overlap_ratio": 0.4}.'
        ),
    )
    parser.add_argument(
        "--imagery-dir", dest="imagery_dir", metavar="DIR",
        help="Ordner mit den Sentinel-2-GeoTIFFs (ein Bild pro Mine).",
    )
    parser.add_argument(
        "--labels-path", dest="labels_path", metavar="PATH",
        help="GeoPackage/Shapefile mit den digitalisierten Dump-Polygonen.",
    )
    parser.add_argument(
        "--mine-id-field", dest="mine_id_field", metavar="SPALTE",
        help="Spaltenname in labels_path, der die Minen-ID enthält.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", metavar="DIR",
        help="Ausgabeordner für das klassifizierte GeoPackage.",
    )
    parser.add_argument(
        "--n-segments-per-mine", dest="n_segments_per_mine", type=int, metavar="N",
        help="SLIC-Zielanzahl Superpixel pro Mine. Wird ignoriert, solange "
             "target_segment_px gesetzt ist (siehe --target-segment-px).",
    )
    parser.add_argument(
        "--target-segment-px", dest="target_segment_px", type=int, metavar="N",
        help="Zielgröße eines Segments in Pixeln; n_segments wird daraus pro "
             "Mine berechnet (gültige Pixel / N), statt eines festen Werts. "
             "0 deaktiviert das und nutzt wieder --n-segments-per-mine.",
    )
    parser.add_argument(
        "--max-segments-per-mine", dest="max_segments_per_mine", type=int, metavar="N",
        help="Obergrenze für die berechnete Segmentanzahl pro Mine (nur mit "
             "--target-segment-px relevant), damit große Minen die Laufzeit "
             "nicht explodieren lassen.",
    )
    parser.add_argument(
        "--compactness", dest="compactness", type=float,
        help="SLIC-Kompaktheit (Form- vs. Farbtreue).",
    )
    parser.add_argument(
        "--min-overlap-ratio", dest="min_overlap_ratio", type=float,
        help="Mindest-Überlappungsanteil, ab dem ein Segment als positiv (Dump) gilt.",
    )
    parser.add_argument(
        "--texture-band", dest="texture_band", metavar="BAND",
        help="Name des Bands, auf dem die GLCM-Textur berechnet wird.",
    )
    parser.add_argument(
        "--n-estimators", dest="n_estimators", type=int, metavar="N",
        help="Anzahl Bäume im Random Forest.",
    )
    parser.add_argument(
        "--random-state", dest="random_state", type=int,
        help="Zufalls-Seed für reproduzierbare Ergebnisse.",
    )
    parser.add_argument(
        "--n-jobs", dest="n_jobs", type=int, metavar="N",
        help="Anzahl paralleler Prozesse zum Verarbeiten der Minen "
             "(jede Mine ist unabhängig von den anderen). "
             "Standard: alle verfügbaren CPU-Kerne. 1 = sequentiell.",
    )
    parser.add_argument(
        "--use-cache", dest="use_cache", action="store_true", default=None,
        help="Zwischengespeicherten Segment-Datensatz aus einem vorherigen "
             "Lauf wiederverwenden (aus output_dir), statt Segmentierung "
             "und Merkmalsberechnung neu zu machen - deutlich schneller "
             "zum Ausprobieren neuer Schwellwerte/Modell-Einstellungen.",
    )
    parser.add_argument(
        "--mine-boundary-path", dest="mine_boundary_path", metavar="PATH",
        help="Optional: Datei mit den tatsächlichen Minen-Grenzen (nicht nur "
             "dem gepufferten Bildausschnitt), um Segmente außerhalb der "
             "Mine auszuschließen. Ohne diese Option kein Filter.",
    )
    parser.add_argument(
        "--mine-boundary-id-field", dest="mine_boundary_id_field", metavar="SPALTE",
        help="Spaltenname für die Minen-ID in --mine-boundary-path.",
    )
    parser.add_argument(
        "--mine-boundary-buffer-m", dest="mine_boundary_buffer_m", type=float, metavar="M",
        help="Zusätzlicher Puffer in Metern um die Minen-Grenze, bevor "
             "gefiltert wird (0 = exakte Grenze).",
    )
    parser.add_argument(
        "--classification-threshold", dest="classification_threshold", type=float, metavar="P",
        help="Schwellwert für dump_proba, ab dem ein Segment als positiv gilt "
             "(0-1, Standard 0.5). Höhere Werte senken False Positives, auf "
             "Kosten der Erkennungsquote - siehe der von der Pipeline "
             "automatisch geloggte Schwellwert-Vergleich.",
    )
    parser.add_argument(
        "--require-neighbor-below", dest="require_neighbor_below", type=float, metavar="P",
        help="Optional: verwirft ein positiv vorhergesagtes Segment mit "
             "dump_proba unter diesem Wert, wenn kein angrenzendes Segment in "
             "derselben Mine ebenfalls positiv ist. Segmente ab diesem Wert "
             "bleiben unangetastet, auch ohne Nachbarn. Zielt auf isolierte "
             "Einzelsegmente mittlerer Konfidenz entlang von Fahrstraßen.",
    )
    parser.add_argument(
        "--require-compact-cluster-below", dest="require_compact_cluster_below", type=float, metavar="P",
        help="Optional: verwirft ein positiv vorhergesagtes Segment mit "
             "dump_proba unter diesem Wert, wenn sein zusammenhängendes "
             "Cluster berührender positiver Segmente eine Kompaktheit unter "
             "--min-cluster-compactness hat (lang und dünn wie eine Straße "
             "statt haufenförmig wie ein Dump). Ergänzt "
             "--require-neighbor-below, statt es zu ersetzen.",
    )
    parser.add_argument(
        "--min-cluster-compactness", dest="min_cluster_compactness", type=float, metavar="P",
        help="Schwellwert für --require-compact-cluster-below (Standard 0.15, "
             "an den echten Daten kalibriert).",
    )
    parser.add_argument(
        "--s1-dir", dest="s1_dir", metavar="DIR",
        help="Optional: Ordner mit Sentinel-1-Radardaten pro Mine (erzeugt von "
             "src/fetch_sentinel1.py). Fügt Radar-Merkmale (VV, VH, VH/VV) pro "
             "Segment hinzu. Ohne diese Option keine Radar-Merkmale.",
    )
    parser.add_argument(
        "--dem-dir", dest="dem_dir", metavar="DIR",
        help="Optional: Ordner mit dem Höhenmodell pro Mine (erzeugt von "
             "src/fetch_dem.py). Fügt Hangneigung, Lage relativ zur Umgebung und "
             "Hangausrichtung pro Segment hinzu.",
    )
    parser.add_argument(
        "--s2t-dir", dest="s2t_dir", metavar="DIR",
        help="Optional: Ordner mit Zeitreihen-Merkmalen aus mehreren "
             "Sentinel-2-Aufnahmen (erzeugt von src/fetch_s2_timeseries.py). "
             "Fügt hinzu, wie stark die Helligkeit übers Jahr schwankt - "
             "Schatten wandern mit dem Sonnenstand, Reifen nicht.",
    )
    parser.add_argument(
        "--s1t-dir", dest="s1t_dir", metavar="DIR",
        help="Optional: Ordner mit Radar-Zeitreihen-Merkmalen aus mehreren "
             "Sentinel-1-Aufnahmen (erzeugt von "
             "src/fetch_sentinel1_timeseries.py). Fügt hinzu, wie stark die "
             "Rückstreuung über mehrere Aufnahmen schwankt - geometrisch klare "
             "Flächen sind blickwinkelabhängig, ein Reifenhaufen streut diffus.",
    )
    parser.add_argument(
        "--extra-imagery-dir", dest="extra_imagery_dir", metavar="DIR",
        help="Nur mit --use-cache: Ordner mit zusätzlichen Kacheln (z.B. einer "
             "neuen Stelle). Diese werden segmentiert, an den Zwischenspeicher "
             "angehängt (trainiert wird nur auf den bekannten Minen) und NUR sie "
             "werden exportiert.",
    )
    parser.add_argument(
        "--extra-labels-path", dest="extra_labels_path", metavar="PATH",
        help="Label-Polygone (mine_id = Kachelname) für --labeled-imagery-dir.",
    )
    parser.add_argument(
        "--labeled-imagery-dir", dest="labeled_imagery_dir", metavar="DIR",
        help="Nur mit --use-cache und --extra-labels-path: Ordner mit Kacheln, deren Dumps "
             "als Label-Polygone vorliegen. Sie werden mittrainiert (anders als "
             "--extra-imagery-dir, das nur bewertet wird).",
    )
    parser.add_argument(
        "--ignore-points-path", dest="ignore_points_path", metavar="PATH",
        help="Punkte bekannter Halden: label=0-Segmente im Umkreis werden aus dem Training genommen.",
    )
    parser.add_argument(
        "--ignore-radius-m", dest="ignore_radius_m", type=float, metavar="M",
        help="Radius der Ignorier-Zone (Standard 150 m).",
    )
    parser.add_argument(
        "--osm-buildings-dir", dest="osm_buildings_dir", metavar="DIR",
        help="Optional: Ordner mit OSM-Gebäudeumrissen pro Mine (erzeugt von "
             "src/fetch_osm_buildings.py). Fügt der exportierten Karte eine "
             "rein informative is_building-Spalte hinzu (kein Filter, ändert "
             "dump_pred nicht) zum selbst Ein-/Ausblenden in QGIS.",
    )
    parser.add_argument(
        "--osm-poi-dir", dest="osm_poi_dir", metavar="DIR",
        help="Optional: Ordner mit OSM-Punkten/Gewerbeflächen pro Mine (erzeugt "
             "von src/fetch_osm_poi.py). Fügt der exportierten Karte eine rein "
             "informative is_poi-Spalte hinzu (kein Filter, ändert dump_pred "
             "nicht) zum selbst Ein-/Ausblenden in QGIS.",
    )
    parser.add_argument(
        "--osm-roads-dir", dest="osm_roads_dir", metavar="DIR",
        help="Optional: Ordner mit OSM-Straßen pro Mine (erzeugt von "
             "src/fetch_osm_roads.py). Fügt der exportierten Karte die rein "
             "informativen Spalten road_density_150m/is_road_grid hinzu (kein "
             "Filter, ändert dump_pred nicht) zum selbst Ein-/Ausblenden in QGIS.",
    )
    parser.add_argument(
        "--road-density-radius-m", dest="road_density_radius_m", type=float, metavar="M",
        help="Radius (Meter) um jeden Segment-Mittelpunkt für road_density_150m "
             "(Standard 150).",
    )
    parser.add_argument(
        "--road-grid-threshold-m", dest="road_grid_threshold_m", type=float, metavar="M",
        help="Schwellwert für is_road_grid (Standard 500m, an den echten Daten "
             "kalibriert: betrifft 0 von 36 bekannten Dumps).",
    )
    return parser


def resolve_config(base_config, args):
    """Baut die endgültige Konfiguration in dieser Prioritätsreihenfolge
    (niedrigste zuerst): CONFIG-Standardwerte < --config-Datei <
    einzelne Kommandozeilen-Flags."""
    cfg = dict(base_config)

    config_path = getattr(args, "config", None)
    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)
        cfg.update(overrides)

    cli_overrides = {
        "imagery_dir": args.imagery_dir,
        "labels_path": args.labels_path,
        "mine_id_field": args.mine_id_field,
        "output_dir": args.output_dir,
        "n_segments_per_mine": args.n_segments_per_mine,
        "max_segments_per_mine": getattr(args, "max_segments_per_mine", None),
        "compactness": args.compactness,
        "min_overlap_ratio": args.min_overlap_ratio,
        "texture_band": args.texture_band,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "use_cache": args.use_cache,
        "extra_imagery_dir": getattr(args, "extra_imagery_dir", None),
        "extra_labels_path": getattr(args, "extra_labels_path", None),
        "labeled_imagery_dir": getattr(args, "labeled_imagery_dir", None),
        "ignore_points_path": getattr(args, "ignore_points_path", None),
        "ignore_radius_m": getattr(args, "ignore_radius_m", None),
        "mine_boundary_path": getattr(args, "mine_boundary_path", None),
        "mine_boundary_id_field": getattr(args, "mine_boundary_id_field", None),
        "mine_boundary_buffer_m": getattr(args, "mine_boundary_buffer_m", None),
        "classification_threshold": getattr(args, "classification_threshold", None),
        "require_neighbor_below": getattr(args, "require_neighbor_below", None),
        "require_compact_cluster_below": getattr(args, "require_compact_cluster_below", None),
        "min_cluster_compactness": getattr(args, "min_cluster_compactness", None),
        "s1_dir": getattr(args, "s1_dir", None),
        "dem_dir": getattr(args, "dem_dir", None),
        "s2t_dir": getattr(args, "s2t_dir", None),
        "s1t_dir": getattr(args, "s1t_dir", None),
        "osm_buildings_dir": getattr(args, "osm_buildings_dir", None),
        "osm_poi_dir": getattr(args, "osm_poi_dir", None),
        "osm_roads_dir": getattr(args, "osm_roads_dir", None),
        "road_density_radius_m": getattr(args, "road_density_radius_m", None),
        "road_grid_threshold_m": getattr(args, "road_grid_threshold_m", None),
    }
    for key, value in cli_overrides.items():
        if value is not None:
            cfg[key] = value

    # --target-segment-px 0 heißt "deaktivieren" (zurück zu einem festen
    # n_segments_per_mine); jeder andere Wert überschreibt normal.
    target_segment_px = getattr(args, "target_segment_px", None)
    if target_segment_px is not None:
        cfg["target_segment_px"] = None if target_segment_px == 0 else target_segment_px

    return cfg
