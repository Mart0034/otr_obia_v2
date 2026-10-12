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
    m_per_px = (x1 - x0) / w_px      # Meter je Bildpixel (Ausschnitt im metrischen KBS)
    geom.transform(to_dest)          # Standort im Ziel-KBS
    ring.transform(to_dest)
    return ms, geom, ring, m_per_px


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


def nice_bar_m(width_m):
    """Länge des Maßstabsbalkens: 10 / 20 / 50 / 100 / 200 m, etwa 15-30 % der Bildbreite."""
    for m in (10, 20, 50, 100, 200, 500):
        if m >= width_m * 0.15:
            return m
    return 500


def paint_scale_bar(image, m_per_px):
    """Weißer Balken mit Beschriftung unten links - die KI soll die Größe der Objekte einschätzen können."""
    from .qtcompat import QFont
    bar_m = nice_bar_m(image.width() * m_per_px)
    w = max(10, int(bar_m / m_per_px))
    x, y = 16, image.height() - 22
    painter = QPainter(image)
    painter.setRenderHint(enum(QPainter, "RenderHint", "Antialiasing"))
    painter.setPen(QPen(QColor("black"), 7))
    painter.drawLine(x, y, x + w, y)
    painter.setPen(QPen(QColor("white"), 3))
    painter.drawLine(x, y, x + w, y)
    font = QFont()
    font.setPixelSize(16)
    font.setBold(True)
    painter.setFont(font)
    for dx, dy, color in ((1, 1, "black"), (0, 0, "white")):
        painter.setPen(QColor(color))
        painter.drawText(x + dx, y - 8 + dy, f"{bar_m} m")
    painter.end()


def paint_overlay(image, ms, site_geom, ring_geom, draw_ring=True, highlight=True, fill=False, m_per_px=None):
    """Standort gelb umrandet (fill=True: zusätzlich halbtransparent gefüllt), gestrichelter Umkreis und
    Maßstabsbalken auf das fertige Bild malen. Standard ohne Füllung: die Füllung verfärbte die Reifen."""
    painter = QPainter(image)
    painter.setRenderHint(enum(QPainter, "RenderHint", "Antialiasing"))
    if highlight:
        pen = QPen(QColor(255, 220, 0))
        pen.setWidth(3)
        painter.setPen(pen)
        if fill:
            _draw_outline(painter, ms, site_geom, pen, QColor(255, 235, 0, 70))
        else:
            painter.setBrush(qt("BrushStyle", "NoBrush"))
            _draw_outline(painter, ms, site_geom, pen)
        painter.setBrush(qt("BrushStyle", "NoBrush"))
    if draw_ring:
        pen = QPen(QColor("#00e5ff"))
        pen.setWidth(3)
        pen.setStyle(qt("PenStyle", "DashLine"))
        painter.setPen(pen)
        painter.setBrush(qt("BrushStyle", "NoBrush"))
        _draw_outline(painter, ms, ring_geom, pen)
    painter.end()
    if m_per_px:
        paint_scale_bar(image, m_per_px)


def render_site(canvas, layer, feature, radius_m, max_px=1024, min_side=MIN_SIDE_M, draw_ring=True):
    """Art 'visible', blockierend: alle sichtbaren Schichten, unabhängig vom aktuellen Kartenausschnitt.
    Gibt (Bild, Meter je Pixel) zurück."""
    ms, geom, ring, m_per_px = _prepare(canvas, layer, feature, radius_m, max_px, min_side, canvas.layers())
    job = QgsMapRendererSequentialJob(ms)
    job.start()
    job.waitForFinished()
    image = job.renderedImage()
    paint_overlay(image, ms, geom, ring, draw_ring=draw_ring, highlight=False, m_per_px=m_per_px)
    return image, m_per_px


def render_selected(canvas, layer, feature, cfg, radius_m):
    """Art 'visible' mit dem Standort als ausgewählt (gelb); die alte Auswahl wird wiederhergestellt.
    Gibt (Bild, Meter je Pixel) zurück."""
    saved = layer.selectedFeatureIds()
    layer.selectByIds([feature.id()])
    try:
        return render_site(canvas, layer, feature, float(radius_m), cfg["max_image_px"],
                           cfg["min_view_m"], cfg["draw_ring"])
    finally:
        layer.selectByIds(saved)


def start_render_clean(canvas, layer, feature, cfg, radius_m, on_done):
    """Art 'clean', im Hintergrund: startet das Rendern und kehrt sofort zurück; on_done(image) wird später
    im Hauptthread aufgerufen: on_done(image, meter_je_pixel) (image = None, wenn es kein sichtbares Satellitenbild gibt)."""
    layers = raster_layers(canvas)
    if not layers:
        on_done(None, None)
        return
    ms, geom, ring, m_per_px = _prepare(canvas, layer, feature, float(radius_m), cfg["max_image_px"],
                                        cfg["min_view_m"], layers)
    job = QgsMapRendererParallelJob(ms)

    def finished():
        _JOBS.discard(job)
        image = job.renderedImage()
        job.deleteLater()
        paint_overlay(image, ms, geom, ring, draw_ring=cfg["draw_ring"], highlight=True,
                      fill=cfg.get("highlight_fill", False), m_per_px=m_per_px)
        on_done(image, m_per_px)

    job.finished.connect(finished)
    _JOBS.add(job)
    job.start()


def render_view_clean(canvas, max_px=768):
    """Den gerade sichtbaren Kartenausschnitt (nur Satellitenbild, kein Umriss) mit Maßstabsbalken rendern -
    für Beispielbilder von beliebigen Stellen, die keine Standorte sind. Gibt (Bild, Meter je Pixel) zurück;
    (None, None), wenn kein Satellitenbild sichtbar ist."""
    layers = raster_layers(canvas)
    if not layers:
        return None, None
    base = canvas.mapSettings()
    ms = QgsMapSettings()
    ms.setLayers(layers)
    ms.setDestinationCrs(base.destinationCrs())
    ms.setExtent(base.visibleExtent())
    ext = base.visibleExtent()
    w_px, h_px = image_size(ext.width(), ext.height(), max_px)
    ms.setOutputSize(QSize(w_px, h_px))
    ms.setBackgroundColor(QColor("white"))
    job = QgsMapRendererSequentialJob(ms)
    job.start()
    job.waitForFinished()
    image = job.renderedImage()
    to_metric = QgsCoordinateTransform(base.destinationCrs(), QgsCoordinateReferenceSystem("EPSG:32719"), QgsProject.instance())
    width_m = to_metric.transformBoundingBox(ext).width()
    m_per_px = width_m / w_px
    paint_scale_bar(image, m_per_px)
    return image, m_per_px


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
