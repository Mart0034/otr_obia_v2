"""Standort + Umkreis als Bild rendern (aus ai_classifier.py)."""

from qgis.core import (QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsMapRendererSequentialJob,
                       QgsMapSettings, QgsProject, QgsRectangle)

from .core import MAX_EDGE_PX, MIN_SIDE_M, image_size, padded_box
from .qtcompat import (QBuffer, QByteArray, QColor, QIODevice, QPainter, QPen, QPointF, QPolygonF, QSize,
                       enum, qt)


def metric_crs(layer):
    crs = layer.crs()
    return QgsCoordinateReferenceSystem("EPSG:32719") if crs.isGeographic() else crs   # Nord-Chile: UTM 19S


def render_site(canvas, layer, feature, radius_m, max_px=1024, min_side=MIN_SIDE_M, draw_ring=True):
    """Rendert Standort + Umkreis (radius_m) mit den sichtbaren Schichten, unabhängig vom aktuellen
    Kartenausschnitt; der Umkreis wird gestrichelt eingezeichnet."""
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
    ms.setLayers(canvas.layers())
    ms.setDestinationCrs(dest)
    ms.setExtent(extent)
    ms.setOutputSize(QSize(w_px, h_px))
    ms.setBackgroundColor(QColor("white"))
    job = QgsMapRendererSequentialJob(ms)
    job.start()
    job.waitForFinished()
    image = job.renderedImage()
    if not draw_ring:
        return image
    ring.transform(to_dest)
    painter = QPainter(image)
    painter.setRenderHint(enum(QPainter, "RenderHint", "Antialiasing"))
    pen = QPen(QColor("#00e5ff"))
    pen.setWidth(3)
    pen.setStyle(qt("PenStyle", "DashLine"))
    painter.setPen(pen)
    to_px = ms.mapToPixel()
    for poly in (ring.asMultiPolygon() if ring.isMultipart() else [ring.asPolygon()]):
        if poly:
            pts = [to_px.transform(p) for p in poly[0]]
            painter.drawPolyline(QPolygonF([QPointF(p.x(), p.y()) for p in pts]))
    painter.end()
    return image


def render_selected(canvas, layer, feature, cfg, radius_m):
    """Wie render_site, aber mit dem Standort als ausgewählt (gelb), auch wenn er nicht der gerade im
    Review-Fenster aktive ist (Vorab-/Stapel-Einschätzung); die alte Auswahl wird wiederhergestellt."""
    saved = layer.selectedFeatureIds()
    layer.selectByIds([feature.id()])
    try:
        return render_site(canvas, layer, feature, float(radius_m), cfg["max_image_px"],
                           cfg["min_view_m"], cfg["draw_ring"])
    finally:
        layer.selectByIds(saved)


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
