"""Standortschicht laden, einrichten und um die KI-Spalten ergänzen (aus load_candidate_sites.py)."""

from qgis.core import (QgsAction, QgsCategorizedSymbolRenderer, QgsDefaultValue, QgsEditorWidgetSetup,
                       QgsPalLayerSettings, QgsProject, QgsRendererCategory, QgsSymbol,
                       QgsTextBufferSettings, QgsTextFormat, QgsVectorLayer, QgsVectorLayerSimpleLabeling)

from .core import AI_FIELDS
from .qtcompat import QColor, make_field

SIZE_COLORS = [  # (size_class, Farbe, Beschriftung)
    ("small", "#f2e394", "small (< 2.500 m²)"),
    ("medium", "#f4a259", "medium (2.500 - 10.000 m²)"),
    ("large", "#e4572e", "large (10.000 - 40.000 m²)"),
    ("very_large", "#a1116b", "very large (> 40.000 m²)"),
]


def layer_by_name(name):
    """Schicht nach Namen (auch leicht umbenannte); None, wenn es sie nicht gibt."""
    project = QgsProject.instance()
    found = project.mapLayersByName(name)
    if not found:
        found = [lyr for lyr in project.mapLayers().values()
                 if lyr.name().lower().startswith(name) or f"layername={name}" in lyr.source()]
    return found[0] if found else None


def ensure_drawn_layer(path):
    """Legt die Schicht drawn_polygons in der Datei an, falls sie fehlt. Muss laufen, BEVOR die Datei in
    QGIS geladen ist, sonst ist sie gesperrt."""
    from osgeo import ogr
    ds = ogr.Open(path, 1)
    if ds is None:
        raise ValueError("Datei lässt sich nicht zum Schreiben öffnen (in QGIS noch geladen?)")
    names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
    if "drawn_polygons" not in names:
        srs = ds.GetLayerByName("candidate_sites").GetSpatialRef()
        layer = ds.CreateLayer("drawn_polygons", srs, ogr.wkbPolygon)
        for name in ("kind", "mine_id", "note"):
            layer.CreateField(ogr.FieldDefn(name, ogr.OFTString))
    ds = None  # schreibt und schließt


def remove_old_layers():
    for name in ("candidate_sites", "drawn_polygons"):
        for old in QgsProject.instance().mapLayersByName(name):
            QgsProject.instance().removeMapLayer(old.id())


def style_by_size(layer):
    cats = []
    for value, color, label in SIZE_COLORS:
        sym = QgsSymbol.defaultSymbol(layer.geometryType())
        fill = QColor(color)
        fill.setAlphaF(0.45)
        sym.setColor(fill)
        sym.symbolLayer(0).setStrokeColor(QColor(color).darker(150))
        sym.symbolLayer(0).setStrokeWidth(0.6)
        cats.append(QgsRendererCategory(value, sym, label))
    layer.setRenderer(QgsCategorizedSymbolRenderer("size_class", cats))


def label_with_rank(layer):
    settings = QgsPalLayerSettings()
    settings.fieldName = "rank"
    fmt = QgsTextFormat()
    fmt.setSize(10)
    fmt.setColor(QColor("black"))
    buf = QgsTextBufferSettings()
    buf.setEnabled(True)
    buf.setSize(1.0)
    buf.setColor(QColor("white"))
    fmt.setBuffer(buf)
    settings.setFormat(fmt)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)


def add_maps_action(layer):
    try:                                         # QGIS 3: QgsAction.OpenUrl, QGIS 4: Qgis.AttributeActionType
        from qgis.core import Qgis
        kind = Qgis.AttributeActionType.OpenUrl
    except (ImportError, AttributeError):
        kind = QgsAction.OpenUrl
    action = QgsAction(kind, "In Google Maps öffnen", '[% "maps_url" %]')
    layer.actions().addAction(action)
    layer.actions().setDefaultAction("Canvas", action.id())


def review_dropdown(layer):
    idx = layer.fields().indexOf("review")
    if idx < 0:
        raise ValueError("Spalte 'review' nicht gefunden")
    options = [{"dump": "dump"}, {"clean": "clean"}, {"partial": "partial"}, {"unsure": "unsure"}]
    layer.setEditorWidgetSetup(idx, QgsEditorWidgetSetup("ValueMap", {"map": options}))


def setup_drawn_layer(path):
    drawn = QgsVectorLayer(f"{path}|layername=drawn_polygons", "drawn_polygons", "ogr")
    if not drawn.isValid():
        raise ValueError("Schicht drawn_polygons nicht in der Datei")
    QgsProject.instance().addMapLayer(drawn)
    cats = []
    for value, color in (("dump", "#d7191c"), ("clean", "#1a9641")):
        sym = QgsSymbol.defaultSymbol(drawn.geometryType())
        fill = QColor(color)
        fill.setAlphaF(0.20)
        sym.setColor(fill)
        sym.symbolLayer(0).setStrokeColor(QColor(color))
        sym.symbolLayer(0).setStrokeWidth(1.0)
        cats.append(QgsRendererCategory(value, sym, value))
    drawn.setRenderer(QgsCategorizedSymbolRenderer("kind", cats))
    drawn.setEditorWidgetSetup(drawn.fields().indexOf("kind"),
                               QgsEditorWidgetSetup("ValueMap", {"map": [{"dump": "dump"}, {"clean": "clean"}]}))
    drawn.setDefaultValueDefinition(
        drawn.fields().indexOf("mine_id"),
        QgsDefaultValue('array_first(overlay_intersects(\'candidate_sites\', "mine_id"))'))
    return drawn


def load_candidate_sites(iface, path):
    """Lädt candidate_sites.gpkg samt Zeichenschicht und stylt sie. Gibt (Schicht, Meldungen) zurück;
    ein fehlgeschlagener Einrichtungsschritt verhindert die übrigen nicht."""
    messages = []

    def step(name, func):
        try:
            func()
        except Exception as exc:
            messages.append(f"{name} fehlgeschlagen: {exc}")

    step("alte Schichten entfernen", remove_old_layers)
    step("Zeichen-Schicht anlegen", lambda: ensure_drawn_layer(path))
    layer = QgsVectorLayer(f"{path}|layername=candidate_sites", "candidate_sites", "ogr")
    if not layer.isValid():
        raise ValueError(f"Schicht konnte nicht geladen werden: {path}")
    QgsProject.instance().addMapLayer(layer)
    layer.setDisplayExpression('"site_id" || \'  \' || "size_class" || \'  (\' || round("area_m2") || \' m²)\'')
    step("Farben nach Größenklasse", lambda: style_by_size(layer))
    step("Rang-Beschriftung", lambda: label_with_rank(layer))
    step("Google-Maps-Aktion", lambda: add_maps_action(layer))
    step("review-Auswahlliste", lambda: review_dropdown(layer))
    step("Auswahlfarbe durchscheinend", lambda: iface.mapCanvas().setSelectionColor(QColor(255, 255, 0, 60)))
    step("Zeichen-Schicht drawn_polygons", lambda: setup_drawn_layer(path))
    layer.triggerRepaint()
    iface.mapCanvas().setExtent(layer.extent())
    iface.mapCanvas().refresh()
    if layer.fields().indexOf("context") < 0:
        messages.append("Die Liste hat keine Spalte 'context' (Betrieb/Wüste): die KI nutzt den festen Umkreis. "
                        "Mit src/site_context.py lässt sie sich ergänzen.")
    return layer, messages


def ensure_ai_fields(layer):
    """Legt die ai_*-Spalten an, falls sie fehlen (die Schicht wird dazu in den Bearbeitungsmodus gesetzt)."""
    if not layer.isEditable():
        layer.startEditing()
    missing = [make_field(n, k) for n, k in AI_FIELDS if layer.fields().indexOf(n) < 0]
    for field in missing:                        # QgsVectorLayer kennt nur addAttribute (Einzahl)
        if not layer.addAttribute(field):
            raise RuntimeError(f"Spalte {field.name()} konnte nicht angelegt werden")
    if missing:
        layer.commitChanges(False)
        if not layer.isEditable():
            layer.startEditing()
