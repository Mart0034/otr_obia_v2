"""
Kandidaten-Standorte in QGIS laden und einrichten
=================================================

Ausführen in QGIS: Erweiterungen > Python-Konsole > "Skript öffnen" (Editor-
Symbol) > diese Datei laden > Ausführen (grüner Pfeil). Es öffnet sich ein
Dateidialog; dort die von src/candidate_sites.py erzeugte candidate_sites.gpkg
wählen. Das Skript

  * lädt die Schicht und färbt die Standorte nach Größenklasse
    (small / medium / large / very_large),
  * beschriftet jeden Standort mit seinem Rang,
  * legt die Aktion "In Google Maps öffnen" an (Werkzeug "Objektaktion
    ausführen" / Rechtsklick > Aktionen; die Adresse steht auch in der
    Spalte maps_url),
  * macht die Spalte `review` zu einer Auswahlliste (dump / clean / partial / unsure),
  * macht die Auswahlfarbe durchscheinend (sonst verdeckt das gelbe
    Auswahl-Highlight das Luftbild),
  * zoomt auf alle Standorte.

Zum Eintragen: Schicht markieren > Bearbeitungsmodus (Stift) > in der
Attributtabelle `review` (und gern `note`) setzen > Änderungen speichern.
Die Datei kann danach zurückgegeben werden; daraus werden echte Negativ-
beispiele (clean) und neue Halden (dump) für das nächste Training.

Hinweis: nur gegen die PyQGIS-3-Schnittstelle geschrieben, in dieser
Umgebung nicht in QGIS ausgeführt. Falls ein Schritt fehlschlägt, meldet das
Skript es und macht mit dem nächsten weiter.
"""

from qgis.core import (
    QgsAction, QgsCategorizedSymbolRenderer, QgsEditorWidgetSetup, QgsPalLayerSettings,
    QgsProject, QgsRendererCategory, QgsSymbol, QgsTextBufferSettings, QgsTextFormat,
    QgsVectorLayer, QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QFileDialog

PATH = ""  # leer lassen = Dateidialog; sonst hier den Pfad zur .gpkg eintragen

SIZE_COLORS = [  # (size_class, Farbe, Beschriftung)
    ("small", "#f2e394", "small (< 2.500 m²)"),
    ("medium", "#f4a259", "medium (2.500 - 10.000 m²)"),
    ("large", "#e4572e", "large (10.000 - 40.000 m²)"),
    ("very_large", "#a1116b", "very large (> 40.000 m²)"),
]


def step(name, func):
    try:
        func()
    except Exception as exc:  # ein fehlgeschlagener Schritt soll den Rest nicht verhindern
        print(f"[candidate_sites] {name} fehlgeschlagen: {exc}")


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
    action = QgsAction(QgsAction.OpenUrl, "In Google Maps öffnen", '[% "maps_url" %]')
    layer.actions().addAction(action)
    layer.actions().setDefaultAction("Canvas", action.id())


def soften_selection():
    """Die Auswahlfarbe von QGIS ist standardmäßig deckend gelb und verdeckt das
    Luftbild. Hier nur schwach durchscheinend, damit man darunter noch sieht."""
    iface.mapCanvas().setSelectionColor(QColor(255, 255, 0, 60))  # noqa: F821


def review_dropdown(layer):
    idx = layer.fields().indexOf("review")
    if idx < 0:
        raise ValueError("Spalte 'review' nicht gefunden")
    options = [{"dump": "dump"}, {"clean": "clean"}, {"partial": "partial"}, {"unsure": "unsure"}]
    layer.setEditorWidgetSetup(idx, QgsEditorWidgetSetup("ValueMap", {"map": options}))


path = PATH or QFileDialog.getOpenFileName(None, "candidate_sites.gpkg wählen", "", "GeoPackage (*.gpkg)")[0]
if path:
    layer = QgsVectorLayer(f"{path}|layername=candidate_sites", "candidate_sites", "ogr")
    if not layer.isValid():
        print(f"[candidate_sites] Schicht konnte nicht geladen werden: {path}")
    else:
        QgsProject.instance().addMapLayer(layer)
        layer.setDisplayExpression('"site_id" || \'  \' || "size_class" || \'  (\' || round("area_m2") || \' m²)\'')
        step("Farben nach Größenklasse", lambda: style_by_size(layer))
        step("Rang-Beschriftung", lambda: label_with_rank(layer))
        step("Google-Maps-Aktion", lambda: add_maps_action(layer))
        step("review-Auswahlliste", lambda: review_dropdown(layer))
        step("Auswahlfarbe durchscheinend", soften_selection)
        layer.triggerRepaint()
        iface.mapCanvas().setExtent(layer.extent())  # noqa: F821  (iface gibt es in der QGIS-Konsole)
        iface.mapCanvas().refresh()  # noqa: F821
        print(f"[candidate_sites] {layer.featureCount()} Standorte geladen.")
