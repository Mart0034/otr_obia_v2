"""Standort + Umkreis als Bild rendern.

Zwei Arten:
  clean    (Standard) nur die sichtbaren Rasterschichten (Satellitenbild) + selbst gezeichneter Standort-
           Umriss (gelb) und gestrichelter Umkreis. Läuft im Hintergrund (QgsMapRendererParallelJob): mehrere
           Standorte gleichzeitig, QGIS bleibt bedienbar, und keine Modell-Ergebnisse verdecken den Boden.
  visible  alle sichtbaren Schichten wie im Kartenfenster, der Standort über die Auswahl (gelb) markiert;
           blockiert QGIS, bis das Bild fertig ist.
"""

from qgis.core import (QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsMapLayerType,
                       QgsMapRendererParallelJob, QgsMapRendererSequentialJob, QgsMapSettings, QgsProject,
                       QgsRectangle)

from .core import MAX_EDGE_PX, MIN_SIDE_M, image_size, padded_box
from .qtcompat import (QBuffer, QByteArray, QColor, QIODevice, QPainter, QPen, QPointF, QPolygonF, QSize,
                       enum, qt)

_JOBS = set()                                   # laufende Hintergrund-Jobs (sonst räumt Python sie weg)


def metric_crs(layer):
    crs = layer.crs()
    return QgsCoordinateReferenceSystem("EPSG:32719") if crs.isGeographic() else crs   # Nord-Chile: UTM 19S


def raster_layers(canvas):
    """Die sichtbaren Rasterschichten (Satellitenbild, XYZ-Kacheln ...) in Zeichenreihenfolge."""
    return [lyr for lyr in canvas.layers() if lyr.type() == QgsMapLayerType.RasterLayer]


def _prepare(canvas, layer, feature, radius_m, max_px, min_side, layers):
    """Kartenausschnitt (Standort + Umkreis) und Geometrien im Ziel-KBS; gemeinsamer Teil beider Arten."""
    project = QgsProject.instance()
    metric = metric_crs(layer)
    dest = canvas.mapSettings().destinationCrs()
    geom = feature.geometry()
    geom.transform(QgsCoordinateTransform(layer.crs(), metric, project))
    ring = geom.buffer(radius_m, 16)
    bb = ring.boundingBox()
    x0, y0, x1, y1 = padded_box(bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum(), 0.0, min_side)
    to_dest = QgsCoordinateTransform(metric, dest, project)
    extent = to_dest.transformBoundingBox(QgsRectangle(x0, y0, x1, y1))
    w_px, h_px = image_size(extent.width(), extent.height(), max_px)
    ms = QgsMapSettings()
    ms.setLayers(layers)
    ms.setDestinationCrs(dest)
    ms.setExtent(extent)
    ms.setOutputSize(QSize(w_px, h_px))
    ms.setBackgroundColor(QColor("white"))
    geom.transform(to_dest)          # Standort im Ziel-KBS
    ring.transform(to_dest)
    return ms, geom, ring


def _polys(geometry):
    return geometry.asMultiPolygon() if geometry.isMultipart() else [geometry.asPolygon()]


def _draw_outline(painter, ms, geometry, pen, fill=None):
    to_px = ms.mapToPixel()
    for poly in _polys(geometry):
        if poly:
            pts = QPolygonF([QPointF(p.x(), p.y()) for p in (to_px.transform(q) for q in poly[0])])
            if fill is not None:
                painter.setBrush(fill)
                painter.drawPolygon(pts)
            else:
                painter.drawPolyline(pts)


def paint_overlay(image, ms, site_geom, ring_geom, draw_ring=True, highlight=True):
    """Standort gelb (halbtransparent, mit Rand) und gestrichelter Umkreis auf das fertige Bild malen."""
    painter = QPainter(image)
    painter.setRenderHint(enum(QPainter, "RenderHint", "Antialiasing"))
    if highlight:
        fill = QColor(255, 235, 0, 70)
        pen = QPen(QColor(255, 220, 0))
        pen.setWidth(3)
        painter.setPen(pen)
        _draw_outline(painter, ms, site_geom, pen, fill)
        painter.setBrush(qt("BrushStyle", "NoBrush"))
    if draw_ring:
        pen = QPen(QColor("#00e5ff"))
        pen.setWidth(3)
        pen.setStyle(qt("PenStyle", "DashLine"))
        painter.setPen(pen)
        painter.setBrush(qt("BrushStyle", "NoBrush"))
        _draw_outline(painter, ms, ring_geom, pen)
    painter.end()


def render_site(canvas, layer, feature, radius_m, max_px=1024, min_side=MIN_SIDE_M, draw_ring=True):
    """Art 'visible', blockierend: alle sichtbaren Schichten, unabhängig vom aktuellen Kartenausschnitt."""
    ms, geom, ring = _prepare(canvas, layer, feature, radius_m, max_px, min_side, canvas.layers())
    job = QgsMapRendererSequentialJob(ms)
    job.start()
    job.waitForFinished()
    image = job.renderedImage()
    if draw_ring:
        paint_overlay(image, ms, geom, ring, draw_ring=True, highlight=False)
    return image


def render_selected(canvas, layer, feature, cfg, radius_m):
    """Art 'visible' mit dem Standort als ausgewählt (gelb); die alte Auswahl wird wiederhergestellt."""
    saved = layer.selectedFeatureIds()
    layer.selectByIds([feature.id()])
    try:
        return render_site(canvas, layer, feature, float(radius_m), cfg["max_image_px"],
                           cfg["min_view_m"], cfg["draw_ring"])
    finally:
        layer.selectByIds(saved)


def start_render_clean(canvas, layer, feature, cfg, radius_m, on_done):
    """Art 'clean', im Hintergrund: startet das Rendern und kehrt sofort zurück; on_done(image) wird später
    im Hauptthread aufgerufen (image = None, wenn es kein sichtbares Satellitenbild gibt)."""
    layers = raster_layers(canvas)
    if not layers:
        on_done(None)
        return
    ms, geom, ring = _prepare(canvas, layer, feature, float(radius_m), cfg["max_image_px"], cfg["min_view_m"], layers)
    job = QgsMapRendererParallelJob(ms)

    def finished():
        _JOBS.discard(job)
        image = job.renderedImage()
        job.deleteLater()
        paint_overlay(image, ms, geom, ring, draw_ring=cfg["draw_ring"], highlight=True)
        on_done(image)

    job.finished.connect(finished)
    _JOBS.add(job)
    job.start()


def image_png(image, max_edge=MAX_EDGE_PX):
    """QImage als PNG-Bytes (auf max_edge begrenzt)."""
    if max(image.width(), image.height()) > max_edge:
        smooth = qt("TransformationMode", "SmoothTransformation")
        image = image.scaledToWidth(max_edge, smooth) if image.width() >= image.height() \
            else image.scaledToHeight(max_edge, smooth)
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(enum(QIODevice, "OpenModeFlag", "WriteOnly"))
    image.save(buf, "PNG")
    buf.close()
    return bytes(data)
