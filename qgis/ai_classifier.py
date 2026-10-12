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

    Einschätzen   schickt den aktuell sichtbaren Ausschnitt (mit dem markierten Standort)
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
LOG_PATH = os.path.join(os.path.expanduser("~"), "ai_classifier_log.csv")

LABELS = ("dump", "clean_road", "clean_desert", "clean_town", "clean_mine", "clean_yard",
          "clean_snow", "clean_water", "clean_vegetation", "clean_other", "unsure")

SYSTEM_PROMPT = """You help review candidate sites for a detection project that looks for \
dumps of discarded giant mining tires (OTR tires) in northern Chile, in satellite imagery.

You are shown a screenshot of a GIS map window. A yellow/highlighted polygon, if present, \
marks the candidate site under review; judge what is inside and immediately around it.

What a tire dump usually looks like: a compact area of dark, speckled or ring-like texture \
(individual tires are 2-4 m across), often piles or neat rows, on bare ground near a mine, \
frequently beside a haul road or waste-rock area. Confusers: roads and haul roads, rock/waste \
piles and pit shadows, equipment yards and workshops, towns and buildings, snow patches, \
dark rock or desert colour variation, water, vegetation.

Classify the view with exactly one label from: %s.
Use "dump" only if tires are plausibly visible; use "unsure" if the image resolution is too \
coarse or the view is ambiguous. Also say whether the dump (if any) is inside the highlighted \
polygon, only partly covered, or elsewhere in the view.

Reply with ONLY a JSON object:
{"label": "<label>", "confidence": <0..1>, "dump_location": "inside|partial|nearby|none", \
"reason": "<one or two sentences>"}""" % ", ".join(LABELS)


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
    from qgis.core import QgsApplication, QgsSettings, QgsTask
    from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice, Qt
    from qgis.PyQt.QtWidgets import (QComboBox, QDockWidget, QGridLayout, QInputDialog, QLabel,
                                     QLineEdit, QPushButton, QWidget)
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


def _canvas_png(canvas):
    """Aktuell sichtbaren Kartenausschnitt als PNG-Bytes (auf MAX_EDGE_PX begrenzt)."""
    image = canvas.grab().toImage()
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
        self.go = QPushButton("Einschätzen (aktueller Ausschnitt)")
        self.go.setMinimumHeight(34)
        self.go.clicked.connect(lambda _=False: self.classify())
        grid.addWidget(self.go, 2, 0, 1, 2)
        self.out = QLabel("")
        self.out.setWordWrap(True)
        self.out.setTextInteractionFlags(_qt_enum("TextInteractionFlag", "TextSelectableByMouse"))
        grid.addWidget(self.out, 3, 0, 1, 2)
        note_btn = QPushButton("Als Notiz übernehmen")
        note_btn.clicked.connect(lambda _=False: self.to_note())
        grid.addWidget(note_btn, 4, 0, 1, 2)
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

    def classify(self):
        key = self._key()
        if not key:
            self.out.setText("Kein API-Key.")
            return
        png = _canvas_png(self.iface.mapCanvas())
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
