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
    ⚙             Einstellungen: Modell, max. Tokens, Temperatur, Prompt, Umkreis, Bildgröße,
                  Zeitlimit, Log-Datei (werden gespeichert)
    Als Notiz     übernimmt das Ergebnis in das Notizfeld des Review-Fensters (falls offen)

Läuft review_queue.py im selben Fenster, wird zusätzlich site_id/Rang mitgeloggt.
Jede Anfrage steht in ~/ai_classifier_log.csv, damit sich die KI-Urteile später
mit deinen vergleichen lassen (Spalte human_review wird aus der Schicht gelesen).

Hinweis: gegen die PyQGIS-3-Schnittstelle geschrieben, nicht in QGIS getestet; die
Anfrage-/Antwort-Logik (build_payload, parse_reply, call_api) ist getestet.
"""

import base64
import csv
import hashlib
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
{"label": "<label>", "confidence": <0..1, how sure you are of the label>, \
"dump_probability": <0..1, probability that a tire dump is present in the zone>, \
"dump_location": "inside|partial|nearby|none", "reason": "<one or two sentences>"}""" % ", ".join(LABELS)


DEFAULT_SETTINGS = {
    "model": DEFAULT_MODEL,
    "max_tokens": 400,
    "temperature": None,          # None = API-Standard
    "system_prompt": SYSTEM_PROMPT,
    "radius_m": BUFFER_M,
    "max_image_px": 1024,         # Kantenlänge des gerenderten Bildes
    "min_view_m": MIN_SIDE_M,
    "draw_ring": True,
    "timeout_s": 90,
    "log_path": LOG_PATH,
}
_LIMITS = {"max_tokens": (16, 8192, int), "radius_m": (0, 500, float), "max_image_px": (256, MAX_EDGE_PX, int),
           "min_view_m": (20, 2000, float), "timeout_s": (10, 600, int)}


def clean_settings(raw):
    """Gespeicherte/eingegebene Werte -> gültige Einstellungen (fehlende/kaputte Werte = Standard,
    Zahlen werden auf sinnvolle Grenzen gekürzt, leerer Prompt = Standard-Prompt)."""
    raw = raw if isinstance(raw, dict) else {}
    out = dict(DEFAULT_SETTINGS)
    for key, (lo, hi, typ) in _LIMITS.items():
        try:
            out[key] = typ(max(lo, min(hi, float(raw[key]))))
        except (KeyError, TypeError, ValueError):
            pass
    temp = raw.get("temperature")
    if temp is not None:
        try:
            out["temperature"] = max(0.0, min(1.0, float(temp)))
        except (TypeError, ValueError):
            pass
    for key in ("model", "system_prompt", "log_path"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            out[key] = raw[key].strip() if key != "system_prompt" else raw[key]
    if isinstance(raw.get("draw_ring"), bool):
        out["draw_ring"] = raw["draw_ring"]
    return out


def load_settings(text):
    try:
        return clean_settings(json.loads(text))
    except (TypeError, ValueError):
        return clean_settings({})


def prompt_id(prompt):
    """Kurze Kennung des Prompts fürs Log (zum Vergleichen verschiedener Prompts)."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]


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


def build_payload(image_b64, media_type="image/png", context="", model=DEFAULT_MODEL, max_tokens=400,
                  system=None, temperature=None):
    """Anfrage-Körper für die Messages-API (reine Funktion, testbar)."""
    text = "Classify what is visible in this map view."
    if context.strip():
        text += f"\nContext: {context.strip()}"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system or SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
            {"type": "text", "text": text}]}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


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
    try:
        dump_p = max(0.0, min(1.0, float(data["dump_probability"])))
    except (KeyError, TypeError, ValueError):      # älterer Prompt ohne das Feld
        dump_p = conf if label == "dump" else (1.0 - conf if label.startswith("clean") else 0.5)
    return {"label": label, "confidence": conf, "dump_probability": dump_p,
            "dump_location": str(data.get("dump_location", "")),
            "reason": str(data.get("reason", text.strip() if not data else ""))}


def banner(result):
    """(Text, Farbe) des Ergebnis-Kastens: 'Dump found (93%)' / 'No dump (04%)' - die Prozentzahl ist
    die Wahrscheinlichkeit, dass eine Halde vorhanden ist; ab 50 % gilt 'Dump found'."""
    p = result["dump_probability"]
    text = "Dump found" if p >= 0.5 else "No dump"
    return f"{text} ({round(p * 100):02d}%)", ("#c62828" if p >= 0.5 else "#2e7d32")


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
    from qgis.PyQt.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget,
                                     QDoubleSpinBox, QFormLayout, QGridLayout, QInputDialog, QLabel,
                                     QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout,
                                     QWidget)
    IN_QGIS = True
except ImportError:
    IN_QGIS = False
    QDockWidget = QDialog = object

SETTINGS_KEY = "otr_obia/anthropic_api_key"
PREFS_KEY = "otr_obia/ai_classifier_prefs"
MODELS = [DEFAULT_MODEL, "claude-opus-5-5", "claude-fable-5-1", "claude-haiku-5-5"]


def _qt_enum(group_name, name):
    group = getattr(Qt, group_name, None)
    if group is not None and hasattr(group, name):
        return getattr(group, name)
    return getattr(Qt, name)


def _metric_crs(layer):
    crs = layer.crs()
    return QgsCoordinateReferenceSystem("EPSG:32719") if crs.isGeographic() else crs   # Nord-Chile: UTM 19S


def _render_site(canvas, layer, feature, radius_m, max_px=1024, min_side=MIN_SIDE_M, draw_ring=True):
    """Rendert Standort + Umkreis (radius_m) mit den sichtbaren Schichten, unabhängig vom
    aktuellen Kartenausschnitt; der Umkreis wird gestrichelt eingezeichnet. None bei Fehler."""
    project = QgsProject.instance()
    metric = _metric_crs(layer)
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


def _image_png(image, max_edge=MAX_EDGE_PX):
    """QImage als PNG-Bytes (auf max_edge begrenzt)."""
    if max(image.width(), image.height()) > max_edge:
        if image.width() >= image.height():
            image = image.scaledToWidth(max_edge, _qt_enum("TransformationMode", "SmoothTransformation"))
        else:
            image = image.scaledToHeight(max_edge, _qt_enum("TransformationMode", "SmoothTransformation"))
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly if hasattr(QIODevice, "OpenModeFlag") else QIODevice.WriteOnly)
    image.save(buf, "PNG")
    buf.close()
    return bytes(data)


class SettingsDialog(QDialog):  # pragma: no cover - braucht QGIS
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle("KI-Einstellungen")
        self.resize(560, 560)
        d = DEFAULT_SETTINGS
        form = QFormLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(MODELS)
        self.model.setEditText(settings["model"])
        form.addRow("Modell", self.model)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(*_LIMITS["max_tokens"][:2])
        self.max_tokens.setValue(settings["max_tokens"])
        form.addRow("Max. Ausgabe-Tokens", self.max_tokens)
        self.use_temp = QCheckBox("Temperatur setzen (sonst API-Standard)")
        self.use_temp.setChecked(settings["temperature"] is not None)
        self.temp = QDoubleSpinBox()
        self.temp.setRange(0.0, 1.0)
        self.temp.setSingleStep(0.1)
        self.temp.setValue(settings["temperature"] if settings["temperature"] is not None else 0.0)
        self.temp.setEnabled(self.use_temp.isChecked())
        self.use_temp.toggled.connect(self.temp.setEnabled)
        form.addRow(self.use_temp, self.temp)
        self.radius = QSpinBox()
        self.radius.setRange(0, 500)
        self.radius.setValue(int(settings["radius_m"]))
        self.radius.setSuffix(" m")
        form.addRow("Umkreis", self.radius)
        self.ring = QCheckBox("Umkreis gestrichelt einzeichnen (Prompt erwähnt die Linie)")
        self.ring.setChecked(settings["draw_ring"])
        form.addRow(self.ring)
        self.px = QSpinBox()
        self.px.setRange(*_LIMITS["max_image_px"][:2])
        self.px.setSingleStep(128)
        self.px.setValue(settings["max_image_px"])
        self.px.setSuffix(" px")
        form.addRow("Bildgröße (lange Kante)", self.px)
        self.min_view = QSpinBox()
        self.min_view.setRange(*[int(v) for v in _LIMITS["min_view_m"][:2]])
        self.min_view.setValue(int(settings["min_view_m"]))
        self.min_view.setSuffix(" m")
        form.addRow("Kleinster Ausschnitt", self.min_view)
        self.timeout = QSpinBox()
        self.timeout.setRange(*_LIMITS["timeout_s"][:2])
        self.timeout.setValue(settings["timeout_s"])
        self.timeout.setSuffix(" s")
        form.addRow("Zeitlimit API", self.timeout)
        self.log_path = QLineEdit(settings["log_path"])
        form.addRow("Log-Datei", self.log_path)
        self.prompt = QPlainTextEdit(settings["system_prompt"])
        reset = QPushButton("Prompt auf Standard zurücksetzen")
        reset.clicked.connect(lambda _=False: self.prompt.setPlainText(d["system_prompt"]))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                                   if hasattr(QDialogButtonBox, "StandardButton")
                                   else QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("System-Prompt (Antwort muss ein JSON mit label / confidence / "
                                "dump_location / reason bleiben):"))
        layout.addWidget(self.prompt, 1)
        layout.addWidget(reset)
        layout.addWidget(buttons)

    def values(self):
        return clean_settings({
            "model": self.model.currentText(), "max_tokens": self.max_tokens.value(),
            "temperature": self.temp.value() if self.use_temp.isChecked() else None,
            "system_prompt": self.prompt.toPlainText(), "radius_m": self.radius.value(),
            "max_image_px": self.px.value(), "min_view_m": self.min_view.value(),
            "draw_ring": self.ring.isChecked(), "timeout_s": self.timeout.value(),
            "log_path": self.log_path.text()})


class AiPanel(QDockWidget):  # pragma: no cover - braucht QGIS
    def __init__(self, iface):
        super().__init__("KI", iface.mainWindow())
        self.iface = iface
        self.last_result = None
        self.task = None
        body = QWidget()
        grid = QGridLayout(body)
        self.settings = load_settings(QgsSettings().value(PREFS_KEY, ""))
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(MODELS)
        self.model.setEditText(self.settings["model"])
        grid.addWidget(QLabel("Modell"), 0, 0)
        grid.addWidget(self.model, 0, 1)
        self.context = QLineEdit()
        self.context.setPlaceholderText("Zusatzinfo (z.B. Sentinel-2 10 m / Satellitenbild 0.5 m)")
        grid.addWidget(self.context, 1, 0, 1, 2)
        self.radius = QSpinBox()
        self.radius.setRange(0, 500)
        self.radius.setValue(int(self.settings["radius_m"]))
        self.radius.setSuffix(" m Umkreis")
        grid.addWidget(self.radius, 2, 0, 1, 2)
        self.go = QPushButton("Einschätzen (Standort + Umkreis)")
        self.go.setMinimumHeight(34)
        self.go.clicked.connect(lambda _=False: self.classify())
        grid.addWidget(self.go, 3, 0, 1, 2)
        self.banner = QLabel("")
        self.banner.setAlignment(_qt_enum("AlignmentFlag", "AlignCenter"))
        self.banner.setMinimumHeight(48)
        self.banner.hide()
        grid.addWidget(self.banner, 4, 0, 1, 2)
        self.out = QLabel("")
        self.out.setWordWrap(True)
        self.out.setTextInteractionFlags(_qt_enum("TextInteractionFlag", "TextSelectableByMouse"))
        grid.addWidget(self.out, 5, 0, 1, 2)
        note_btn = QPushButton("Als Notiz übernehmen")
        note_btn.clicked.connect(lambda _=False: self.to_note())
        grid.addWidget(note_btn, 6, 0, 1, 2)
        gear = QPushButton("⚙ Einstellungen")
        gear.clicked.connect(lambda _=False: self.open_settings())
        grid.addWidget(gear, 7, 0, 1, 2)
        self.setWidget(body)
        iface.addDockWidget(_qt_enum("DockWidgetArea", "RightDockWidgetArea"), self)

    def _store(self):
        QgsSettings().setValue(PREFS_KEY, json.dumps(self.settings))

    def open_settings(self):
        self.settings["model"] = self.model.currentText().strip() or DEFAULT_MODEL
        self.settings["radius_m"] = float(self.radius.value())
        dlg = SettingsDialog(self, clean_settings(self.settings))
        if dlg.exec():
            self.settings = dlg.values()
            self.model.setEditText(self.settings["model"])
            self.radius.setValue(int(self.settings["radius_m"]))
            self._store()
            self.out.setText("Einstellungen gespeichert.")

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
        cfg = self.settings = clean_settings({**self.settings, "model": self.model.currentText(),
                                              "radius_m": self.radius.value()})
        self._store()
        canvas = self.iface.mapCanvas()
        layer, feature = self._target()
        image = None
        if feature is not None and self.radius.value() > 0:
            image = _render_site(canvas, layer, feature, float(self.radius.value()), cfg["max_image_px"],
                                 cfg["min_view_m"], cfg["draw_ring"])
        if image is None:                      # kein Standort gewählt: sichtbaren Ausschnitt nehmen
            image = canvas.grab().toImage()
        png = _image_png(image, cfg["max_image_px"])
        payload = build_payload(base64.b64encode(png).decode("ascii"), "image/png",
                                self.context.text(), cfg["model"], cfg["max_tokens"],
                                cfg["system_prompt"], cfg["temperature"])
        info = self._site_info()
        self.go.setEnabled(False)
        self.banner.hide()
        self.out.setText("Frage Claude ...")

        def work(task):
            return call_api(payload, key, timeout=cfg["timeout_s"])

        def done(exception, result=None):
            self.go.setEnabled(True)
            if exception is not None or result is None:
                self.banner.hide()
                self.out.setText(f"Fehler: {exception}")
                return
            res = parse_reply(result)
            self.last_result = res
            text, color = banner(res)
            self.banner.setText(text)
            self.banner.setStyleSheet(f"background: {color}; color: white; font-size: 20px; "
                                      "font-weight: bold; border-radius: 6px; padding: 6px;")
            self.banner.show()
            self.out.setText(f"<b>{res['label']}</b> &nbsp; ({res['confidence']:.2f})"
                             f"<br>Halde: {res['dump_location'] or '-'}<br>{res['reason']}")
            try:
                log_row(cfg["log_path"], {**info, "model": payload["model"], "prompt_id": prompt_id(cfg["system_prompt"]), "label": res["label"],
                                   "confidence": res["confidence"], "dump_probability": res["dump_probability"], "dump_location": res["dump_location"],
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
