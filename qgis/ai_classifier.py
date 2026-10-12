"""
KI-Einschätzung des sichtbaren Kartenausschnitts in QGIS (Test)
================================================================

Macht einen Screenshot des QGIS-Kartenfensters, schickt ihn mit deinem Anthropic-
API-Key an Claude und zeigt die Einschätzung an: dump / clean + Art (road, desert,
town, ...), Sicherheit und eine kurze Begründung.

Benutzung: in der QGIS-Python-Konsole öffnen und ausführen (wie review_queue.py).
Rechts erscheint das Fenster "KI". Der Key wird aus der Umgebungsvariable
ANTHROPIC_API_KEY gelesen; sonst fragt das Fenster einmal danach und merkt ihn sich
in den QGIS-Einstellungen (Klartext im Profil - auf geteilten Rechnern nicht nutzen).

    Einschätzen   rendert den aktuellen Standort (Review-Fenster bzw. in candidate_sites
                  markiert) plus Umkreis (Standard 50 m, einstellbar), zeichnet den Umkreis
                  gestrichelt ein und schickt das Bild; ohne Standort den sichtbaren Ausschnitt
    Zusatzinfo    optionaler Hinweis, z.B. "Sentinel-2, 10 m" oder "Satellitenbild 0.5 m"
    Als Notiz     übernimmt das Ergebnis in das Notizfeld des Review-Fensters (falls offen)

Läuft review_queue.py im selben Fenster, wird zusätzlich site_id/Rang mitgeloggt.
Jede Anfrage steht in ~/ai_classifier_log.csv, damit sich die KI-Urteile später
mit deinen vergleichen lassen (Spalte human_review wird aus der Schicht gelesen).

Hinweis: gegen die PyQGIS-3-Schnittstelle geschrieben, nicht in QGIS getestet; die
Anfrage-/Antwort-Logik (build_payload, parse_reply, call_api) ist getestet.
"""

import base64
import csv
import json
import os
import re
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_EDGE_PX = 1568          # größere Bilder verkleinert die API ohnehin
BUFFER_M = 50.0             # Umkreis um den Standort, der mitbeurteilt wird
MIN_SIDE_M = 150.0          # kleinster Ausschnitt (Meter), damit winzige Standorte sichtbar bleiben
LOG_PATH = os.path.join(os.path.expanduser("~"), "ai_classifier_log.csv")

LABELS = ("dump", "clean_road", "clean_desert", "clean_town", "clean_mine", "clean_yard",
          "clean_snow", "clean_water", "clean_vegetation", "clean_other", "unsure")

SYSTEM_PROMPT = """You help review candidate sites for a detection project that looks for \
dumps of discarded giant mining tires (OTR tires) in northern Chile, in satellite imagery.

You are shown a screenshot of a GIS map. A yellow/highlighted polygon, if present, \
marks the candidate site under review. A dashed cyan line, if present, marks the zone \
around the polygon (normally 50 m). Judge everything inside the dashed line: the polygon \
itself and its surroundings. Ignore what lies outside the dashed line.

What a tire dump usually looks like: a compact area of dark, speckled or ring-like texture \
(individual tires are 2-4 m across), often piles or neat rows, on bare ground near a mine, \
frequently beside a haul road or waste-rock area. Confusers: roads and haul roads, rock/waste \
piles and pit shadows, equipment yards and workshops, towns and buildings, snow patches, \
dark rock or desert colour variation, water, vegetation.

Classify the view with exactly one label from: %s.
Use "dump" only if tires are plausibly visible; use "unsure" if the image resolution is too \
coarse or the view is ambiguous. Also say where the dump (if any) is: "inside" the highlighted \
polygon, "partial" (polygon only partly covers it), "nearby" (inside the dashed zone but outside \
the polygon), or "none".

Reply with ONLY a JSON object:
{"label": "<label>", "confidence": <0..1>, "dump_location": "inside|partial|nearby|none", \
"reason": "<one or two sentences>"}""" % ", ".join(LABELS)


def padded_box(xmin, ymin, xmax, ymax, radius=BUFFER_M, min_side=MIN_SIDE_M):
    """Umschließendes Rechteck + Umkreis; mindestens min_side Meter pro Seite (um die Mitte)."""
    xmin, ymin, xmax, ymax = xmin - radius, ymin - radius, xmax + radius, ymax + radius
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half_w, half_h = max(xmax - xmin, min_side) / 2, max(ymax - ymin, min_side) / 2
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def image_size(width_m, height_m, max_px=1024):
    """Pixelgröße mit gleichem Seitenverhältnis, längste Seite = max_px."""
    if width_m >= height_m:
        return max_px, max(1, round(max_px * height_m / width_m))
    return max(1, round(max_px * width_m / height_m)), max_px


def build_payload(image_b64, media_type="image/png", context="", model=DEFAULT_MODEL, max_tokens=400):
    """Anfrage-Körper für die Messages-API (reine Funktion, testbar)."""
    text = "Classify what is visible in this map view."
    if context.strip():
        text += f"\nContext: {context.strip()}"
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
            {"type": "text", "text": text}]}],
    }


def parse_reply(response):
    """Messages-API-Antwort -> {'label','confidence','dump_location','reason'}.
    Toleriert Text vor/nach dem JSON (und Code-Zäune); bei Müll label='unsure'."""
    text = "".join(b.get("text", "") for b in response.get("content", []) if b.get("type") == "text")
    match = re.search(r"\{.*\}", text, re.S)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except ValueError:
            data = {}
    label = str(data.get("label", "unsure")).strip().lower()
    if label not in LABELS:
        label = "unsure"
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"label": label, "confidence": conf,
            "dump_location": str(data.get("dump_location", "")),
            "reason": str(data.get("reason", text.strip() if not data else ""))}


def call_api(payload, api_key, timeout=90, opener=urllib.request.urlopen):
    """POST an die Messages-API; wirft RuntimeError mit lesbarer Meldung."""
    req = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"content-type": "application/json", "x-api-key": api_key,
                 "anthropic-version": API_VERSION})
    try:
        with opener(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            detail = ""
        raise RuntimeError(f"API-Fehler {exc.code}: {detail or exc.reason}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Keine Verbindung zur API: {exc.reason}")


def log_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


try:  # außerhalb von QGIS (Tests) nicht verfügbar
    from qgis.core import (QgsApplication, QgsCoordinateReferenceSystem, QgsCoordinateTransform,
                           QgsMapRendererSequentialJob, QgsMapSettings, QgsProject, QgsRectangle,
                           QgsSettings, QgsTask)
    from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QSize, Qt
    from qgis.PyQt.QtGui import QColor, QPainter, QPen, QPolygonF
    from qgis.PyQt.QtWidgets import (QComboBox, QDockWidget, QGridLayout, QInputDialog, QLabel,
                                     QLineEdit, QPushButton, QSpinBox, QWidget)
    IN_QGIS = True
except ImportError:
    IN_QGIS = False
    QDockWidget = object

SETTINGS_KEY = "otr_obia/anthropic_api_key"
MODELS = [DEFAULT_MODEL, "claude-opus-5-5", "claude-fable-5-1", "claude-haiku-5-5"]


def _qt_enum(group_name, name):
    group = getattr(Qt, group_name, None)
    if group is not None and hasattr(group, name):
        return getattr(group, name)
    return getattr(Qt, name)


def _metric_crs(layer):
    crs = layer.crs()
    return QgsCoordinateReferenceSystem("EPSG:32719") if crs.isGeographic() else crs   # Nord-Chile: UTM 19S


def _render_site(canvas, layer, feature, radius_m):
    """Rendert Standort + Umkreis (radius_m) mit den sichtbaren Schichten, unabhängig vom
    aktuellen Kartenausschnitt; der Umkreis wird gestrichelt eingezeichnet. None bei Fehler."""
    project = QgsProject.instance()
    metric = _metric_crs(layer)
    dest = canvas.mapSettings().destinationCrs()
    geom = feature.geometry()
    geom.transform(QgsCoordinateTransform(layer.crs(), metric, project))
    ring = geom.buffer(radius_m, 16)
    bb = ring.boundingBox()
    x0, y0, x1, y1 = padded_box(bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum(), 0.0)
    to_dest = QgsCoordinateTransform(metric, dest, project)
    extent = to_dest.transformBoundingBox(QgsRectangle(x0, y0, x1, y1))
    w_px, h_px = image_size(extent.width(), extent.height(), 1024)

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

    ring.transform(to_dest)
    painter = QPainter(image)
    painter.setRenderHint(getattr(QPainter, "RenderHint", QPainter).Antialiasing)
    pen = QPen(QColor("#00e5ff"))
    pen.setWidth(3)
    pen.setStyle(_qt_enum("PenStyle", "DashLine"))
    painter.setPen(pen)
    to_px = ms.mapToPixel()
    for poly in (ring.asMultiPolygon() if ring.isMultipart() else [ring.asPolygon()]):
        if poly:
            pts = [to_px.transform(p) for p in poly[0]]
            painter.drawPolyline(QPolygonF([QPointF(p.x(), p.y()) for p in pts]))
    painter.end()
    return image


def _image_png(image):
    """QImage als PNG-Bytes (auf MAX_EDGE_PX begrenzt)."""
    if max(image.width(), image.height()) > MAX_EDGE_PX:
        if image.width() >= image.height():
            image = image.scaledToWidth(MAX_EDGE_PX, _qt_enum("TransformationMode", "SmoothTransformation"))
        else:
            image = image.scaledToHeight(MAX_EDGE_PX, _qt_enum("TransformationMode", "SmoothTransformation"))
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly if hasattr(QIODevice, "OpenModeFlag") else QIODevice.WriteOnly)
    image.save(buf, "PNG")
    buf.close()
    return bytes(data)


class AiPanel(QDockWidget):  # pragma: no cover - braucht QGIS
    def __init__(self, iface):
        super().__init__("KI", iface.mainWindow())
        self.iface = iface
        self.last_result = None
        self.task = None
        body = QWidget()
        grid = QGridLayout(body)
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(MODELS)
        grid.addWidget(QLabel("Modell"), 0, 0)
        grid.addWidget(self.model, 0, 1)
        self.context = QLineEdit()
        self.context.setPlaceholderText("Zusatzinfo (z.B. Sentinel-2 10 m / Satellitenbild 0.5 m)")
        grid.addWidget(self.context, 1, 0, 1, 2)
        self.radius = QSpinBox()
        self.radius.setRange(0, 500)
        self.radius.setValue(int(BUFFER_M))
        self.radius.setSuffix(" m Umkreis")
        grid.addWidget(self.radius, 2, 0, 1, 2)
        self.go = QPushButton("Einschätzen (Standort + Umkreis)")
        self.go.setMinimumHeight(34)
        self.go.clicked.connect(lambda _=False: self.classify())
        grid.addWidget(self.go, 3, 0, 1, 2)
        self.out = QLabel("")
        self.out.setWordWrap(True)
        self.out.setTextInteractionFlags(_qt_enum("TextInteractionFlag", "TextSelectableByMouse"))
        grid.addWidget(self.out, 4, 0, 1, 2)
        note_btn = QPushButton("Als Notiz übernehmen")
        note_btn.clicked.connect(lambda _=False: self.to_note())
        grid.addWidget(note_btn, 5, 0, 1, 2)
        self.setWidget(body)
        iface.addDockWidget(_qt_enum("DockWidgetArea", "RightDockWidgetArea"), self)

    def _key(self):
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        settings = QgsSettings()
        if not key:
            key = str(settings.value(SETTINGS_KEY, "") or "").strip()
        if not key:
            echo = QLineEdit.EchoMode.Password if hasattr(QLineEdit, "EchoMode") else QLineEdit.Password
            key, ok = QInputDialog.getText(self, "Anthropic API-Key", "API-Key (sk-ant-...):", echo)
            key = key.strip() if ok else ""
            if key:
                settings.setValue(SETTINGS_KEY, key)
        return key

    def _site_info(self):
        """site_id/Rang/Urteil aus dem Review-Fenster, falls es läuft."""
        panel = globals().get("_review_panel")
        try:
            fid = panel.queue.current
            f = panel.feats[fid]
            return {"site_id": f["site_id"], "rank": f["rank"]}
        except Exception:
            return {"site_id": "", "rank": ""}

    def _target(self):
        """(Schicht, Feature) des aktuellen Standorts: Review-Fenster, sonst Auswahl in candidate_sites."""
        panel = globals().get("_review_panel")
        try:
            return panel.layer, panel.feats[panel.queue.current]
        except Exception:
            pass
        for layer in QgsProject.instance().mapLayersByName("candidate_sites"):
            sel = layer.selectedFeatures()
            if sel:
                return layer, sel[0]
        return None, None

    def classify(self):
        key = self._key()
        if not key:
            self.out.setText("Kein API-Key.")
            return
        canvas = self.iface.mapCanvas()
        layer, feature = self._target()
        image = None
        if feature is not None and self.radius.value() > 0:
            image = _render_site(canvas, layer, feature, float(self.radius.value()))
        if image is None:                      # kein Standort gewählt: sichtbaren Ausschnitt nehmen
            image = canvas.grab().toImage()
        png = _image_png(image)
        payload = build_payload(base64.b64encode(png).decode("ascii"), "image/png",
                                self.context.text(), self.model.currentText().strip() or DEFAULT_MODEL)
        info = self._site_info()
        self.go.setEnabled(False)
        self.out.setText("Frage Claude ...")

        def work(task):
            return call_api(payload, key)

        def done(exception, result=None):
            self.go.setEnabled(True)
            if exception is not None or result is None:
                self.out.setText(f"Fehler: {exception}")
                return
            res = parse_reply(result)
            self.last_result = res
            self.out.setText(f"<b>{res['label']}</b> &nbsp; ({res['confidence']:.2f})"
                             f"<br>Halde: {res['dump_location'] or '-'}<br>{res['reason']}")
            try:
                log_row(LOG_PATH, {**info, "model": payload["model"], "label": res["label"],
                                   "confidence": res["confidence"], "dump_location": res["dump_location"],
                                   "reason": res["reason"]})
            except OSError:
                pass

        self.task = QgsTask.fromFunction("Claude Einschätzung", work, on_finished=done)
        QgsApplication.taskManager().addTask(self.task)

    def to_note(self):
        panel = globals().get("_review_panel")
        if not self.last_result or panel is None:
            return
        panel.note.setEditText(f"AI: {self.last_result['label']} ({self.last_result['confidence']:.2f})")


if IN_QGIS:
    try:
        _ai_panel.close()  # noqa: F821  (altes Fenster einer früheren Ausführung)
        _ai_panel.deleteLater()  # noqa: F821
    except Exception:
        pass
    try:  # Verbindung zum Review-Fenster aus review_queue.py (Variable _panel in der Konsole)
        _review_panel = _panel  # noqa: F821
    except NameError:
        _review_panel = None
    _ai_panel = AiPanel(iface)  # noqa: F821
