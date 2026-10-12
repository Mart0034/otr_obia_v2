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
    Auto          Haken "Auto": der aktuelle Standort wird sofort eingeschätzt, dazu (Einstellung
                  "Vorab einschätzen", Standard 4) die nächsten Standorte der Warteschlange schon im
                  Voraus und parallel - bis du dort bist, liegt das Ergebnis meist fertig vor.
                  Nach Ablehnen wartet der Standort auf dein Urteil im Review-Fenster
    ✔ / ✖         Annehmen schreibt das KI-Urteil als `review` (Dump -> dump bzw. partial, Halde
                  nur in der Umgebung -> unsure, sonst clean) ins gpkg und springt im Review-
                  Fenster weiter; Ablehnen lässt `review` leer. Beides landet in den neuen
                  Spalten ai_label, ai_dump_p, ai_decision (accepted/denied).
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
import time
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
"dump_location": "inside|partial|nearby|none", "reason": "<max 20 words>"}

The reason must be telegraphic: only what is visible and why it decides the label, e.g. \
"Open-pit terraces, dark waste-rock slopes, no tire patterns." Never start with filler such as \
"The zone shows", "The image shows", "This area contains" or "I can see".""" % ", ".join(LABELS)


# prompt_id() früherer Standard-Prompts (vor dem Kürzen der Begründung)
LEGACY_PROMPT_IDS = {"1a72a531", "af179aa7", "bb221802"}

EFFORTS = ("", "low", "medium", "high", "xhigh", "max")

DEFAULT_SETTINGS = {
    "model": DEFAULT_MODEL,
    "max_tokens": 2048,           # Thinking zählt mit: zu klein = Antwort bleibt leer
    "effort": "",                 # "" = API-Standard des Modells; low / medium / high / xhigh / max
    "thinking": "default",        # default | off (off = möglichst wenig/kein Thinking)
    "temperature": None,          # None = API-Standard
    "system_prompt": SYSTEM_PROMPT,
    "radius_m": BUFFER_M,
    "max_image_px": 1024,         # Kantenlänge des gerenderten Bildes
    "min_view_m": MIN_SIDE_M,
    "draw_ring": True,
    "timeout_s": 90,
    "log_path": LOG_PATH,
    "price_in": 0.0,              # USD je Mio. Eingabe-Token; 0 = Preistabelle des Modells
    "price_out": 0.0,
    "prefetch": 4,                # so viele Standorte vor dem aktuellen werden schon vorab eingeschätzt (Auto)
    "auto_next": False,           # nach Annehmen/Ablehnen den nächsten Standort automatisch einschätzen
}
_LIMITS = {"max_tokens": (16, 32000, int), "radius_m": (0, 500, float), "max_image_px": (256, MAX_EDGE_PX, int),
           "min_view_m": (20, 2000, float), "timeout_s": (10, 600, int), "prefetch": (0, 16, int)}


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
    if prompt_id(out["system_prompt"]) in LEGACY_PROMPT_IDS:   # früherer Standard-Prompt -> neuer Standard
        out["system_prompt"] = SYSTEM_PROMPT
    for key in ("price_in", "price_out"):
        try:
            out[key] = max(0.0, min(1000.0, float(raw[key])))
        except (KeyError, TypeError, ValueError):
            pass
    if raw.get("effort") in EFFORTS:
        out["effort"] = raw["effort"]
    if raw.get("thinking") in ("default", "off"):
        out["thinking"] = raw["thinking"]
    for key in ("draw_ring", "auto_next"):
        if isinstance(raw.get(key), bool):
            out[key] = raw[key]
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


def thinking_param(model, mode, effort=""):
    """Wert für `thinking` in der Anfrage. "default" = weglassen (die Claude-5-Modelle denken dann
    selbst, und das Thinking zählt zu max_tokens). "off": Sonnet 5.5 kennt nur "between_tools" (kein
    Thinking vorab), die übrigen "disabled"; Opus 5.5 lässt sich nicht abschalten, und bei effort
    xhigh/max lehnt die API das Abschalten ab - dann wird `thinking` weggelassen."""
    if mode != "off" or effort in ("xhigh", "max") or "opus-5-5" in model:
        return None
    return {"type": "between_tools"} if "sonnet-5-5" in model else {"type": "disabled"}


def build_payload(image_b64, media_type="image/png", context="", model=DEFAULT_MODEL, max_tokens=400,
                  system=None, temperature=None, effort="", thinking="default"):
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
    if effort:
        payload["output_config"] = {"effort": effort}
    think = thinking_param(model, thinking, effort)
    if think is not None:
        payload["thinking"] = think
    return payload


_FILLER = re.compile(r"^\s*(?:the|this)\s+(?:zone|area|image|view|site|scene|polygon|map|region)"
                     r"(?:\s+\w+)?\s+(?:shows?|contains?|displays?|appears?\s+to\s+(?:show|contain)|"
                     r"looks?\s+like|is\s+showing|depicts?)\s*[:,]?\s*", re.I)


def tidy_reason(text):
    """Füllwörter am Anfang ('The zone shows ...') entfernen und den Satz groß beginnen lassen -
    auch wenn ein älterer gespeicherter Prompt sie noch erzeugt."""
    text = " ".join(str(text).split())
    cut = _FILLER.sub("", text)
    return (cut[:1].upper() + cut[1:]) if cut != text and cut else text


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
    return {"ok": bool(data), "stop_reason": response.get("stop_reason", ""),
            "tokens_in": usage_of(response)[0], "tokens_out": usage_of(response)[1],
            "label": label, "confidence": conf, "dump_probability": dump_p,
            "dump_location": str(data.get("dump_location", "")),
            "reason": tidy_reason(data.get("reason", text.strip() if not data else ""))}


def _call_retrying(payload, api_key, timeout, opener, sleep, tries=4):
    """call_api mit Wiederholung bei Überlastung/Ratenlimit (429, 529, 503): 2, 4, 8 s Pause."""
    for attempt in range(tries):
        try:
            return call_api(payload, api_key, timeout, opener)
        except RuntimeError as exc:
            if attempt == tries - 1 or not re.match(r"API-Fehler (429|503|529)", str(exc)):
                raise
            sleep(2 ** (attempt + 1))


# USD je 1 Mio. Token (Eingabe, Ausgabe) laut Anthropic-Preisseite; Thinking zählt als Ausgabe.
# Reihenfolge: spezifischere Namen zuerst. Preise können sich ändern - in den Einstellungen überschreibbar.
PRICES = (("haiku-5-5", 0.10, 0.50), ("haiku-4-5", 1.0, 5.0), ("sonnet-5-5", 2.0, 10.0),
          ("sonnet-5", 2.0, 10.0), ("sonnet-4-6", 3.0, 15.0), ("opus-5-5", 4.0, 20.0),
          ("opus-5", 5.0, 25.0), ("opus-4", 5.0, 25.0), ("fable-5", 10.0, 50.0))


def price_per_mtok(model, override_in=0.0, override_out=0.0):
    """(Eingabe, Ausgabe) in USD je Mio. Token; override > 0 ersetzt den Tabellenwert; None = unbekannt."""
    if override_in > 0 and override_out > 0:
        return override_in, override_out
    for name, p_in, p_out in PRICES:
        if name in model:
            return p_in, p_out
    return None


def usage_of(response):
    """(Eingabe-, Ausgabe-Token) einer Antwort; bei wiederholter Anfrage die Summe beider."""
    u = response.get("usage") or {}
    inp, out = int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)
    extra = response.get("_usage_before") or (0, 0)
    return inp + extra[0], out + extra[1]


def estimate_cost(model, tokens_in, tokens_out, override_in=0.0, override_out=0.0):
    """Geschätzte Kosten in USD (None, wenn der Preis des Modells unbekannt ist)."""
    price = price_per_mtok(model, override_in, override_out)
    if price is None:
        return None
    return (tokens_in * price[0] + tokens_out * price[1]) / 1e6


def fmt_cost(usd):
    if usd is None:
        return "Preis unbekannt"
    return f"≈ ${usd:.4f}" if usd < 0.01 else f"≈ ${usd:.3f}"


def run_check(payload, api_key, timeout=90, opener=urllib.request.urlopen, sleep=time.sleep):
    """Anfrage senden; endet sie wegen max_tokens ohne lesbare Antwort (Thinking hat die Tokens
    verbraucht), einmal mit dem 4-fachen Limit (höchstens 32000) wiederholen."""
    resp = _call_retrying(payload, api_key, timeout, opener, sleep)
    if not parse_reply(resp)["ok"] and resp.get("stop_reason") == "max_tokens":
        bigger = dict(payload, max_tokens=min(32000, payload["max_tokens"] * 4))
        first = usage_of(resp)
        resp = _call_retrying(bigger, api_key, timeout, opener, sleep)
        resp["_retried_with"] = bigger["max_tokens"]
        resp["_usage_before"] = first                  # die erste, abgeschnittene Anfrage wurde auch berechnet
    return resp


def banner(result):
    """(Text, Farbe) des Ergebnis-Kastens: 'Dump found (93%)' / 'No dump (04%)' - die Prozentzahl ist
    die Wahrscheinlichkeit, dass eine Halde vorhanden ist; ab 50 % gilt 'Dump found'."""
    if not result.get("ok", True):
        return "No answer (unreadable)", "#6d6d6d"
    p = result["dump_probability"]
    text = "Dump found" if p >= 0.5 else "No dump"
    return f"{text} ({round(p * 100):02d}%)", ("#c62828" if p >= 0.5 else "#2e7d32")


def ai_note(result):
    """Notiztext: was die KI zu sehen glaubt, z.B. 'AI: clean_road, dump 0.04 - Haul road with ...'."""
    reason = " ".join(str(result.get("reason", "")).split())
    return f"AI: {result['label']}, dump {result['dump_probability']:.2f}" + (f" - {reason}" if reason else "")


def verdict_to_review(result):
    """KI-Ergebnis -> Wert der Spalte `review`: dump / partial / unsure / clean.
    Halde nur teilweise im Umriss = partial; Halde nur in der Umgebung (Beinahe-Treffer) = unsure,
    damit der Umriss nicht fälschlich als 'clean' ins Training geht."""
    if result["dump_probability"] < 0.5:
        return "clean"
    return {"partial": "partial", "nearby": "unsure"}.get(result.get("dump_location", ""), "dump")


AI_FIELDS = (("ai_label", "string"), ("ai_dump_p", "double"), ("ai_decision", "string"))


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
    from qgis.core import (QgsApplication, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsField,
                           QgsMapRendererSequentialJob, QgsMapSettings, QgsProject, QgsRectangle,
                           QgsSettings, QgsTask)
    from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QSize, Qt, QTimer, QVariant
    from qgis.PyQt.QtGui import QColor, QPainter, QPen, QPolygonF
    from qgis.PyQt.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget,
                                     QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QInputDialog, QLabel,
                                     QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QTextBrowser, QVBoxLayout,
                                     QWidget)
    IN_QGIS = True
except ImportError:
    IN_QGIS = False
    QDockWidget = QDialog = QTextBrowser = object

SETTINGS_KEY = "otr_obia/anthropic_api_key"
PREFS_KEY = "otr_obia/ai_classifier_prefs"
MODELS = [DEFAULT_MODEL, "claude-opus-5-5", "claude-fable-5-1", "claude-haiku-5-5"]


def _qt_enum(group_name, name):
    group = getattr(Qt, group_name, None)
    if group is not None and hasattr(group, name):
        return getattr(group, name)
    return getattr(Qt, name)


def _field_type(kind):
    """QVariant (Qt5) bzw. QMetaType (Qt6) für string/double."""
    try:
        from qgis.PyQt.QtCore import QMetaType
        return getattr(QMetaType.Type, "QString" if kind == "string" else "Double")
    except (ImportError, AttributeError):
        return QVariant.String if kind == "string" else QVariant.Double


def _ensure_ai_fields(layer):
    """Legt ai_label / ai_dump_p / ai_decision an, falls sie fehlen (Schicht muss bearbeitbar sein)."""
    if not layer.isEditable():
        layer.startEditing()
    missing = [QgsField(n, _field_type(k)) for n, k in AI_FIELDS if layer.fields().indexOf(n) < 0]
    if missing:
        for field in missing:                  # QgsVectorLayer kennt nur addAttribute (Einzahl)
            if not layer.addAttribute(field):
                raise RuntimeError(f"Spalte {field.name()} konnte nicht angelegt werden")
        layer.commitChanges(False)
        if not layer.isEditable():
            layer.startEditing()


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


def _render_selected(canvas, layer, feature, cfg):
    """Wie _render_site, aber mit dem Standort als ausgewählt (gelb) - auch wenn er nicht der gerade
    im Review-Fenster aktive ist (Vorab-Einschätzung); die alte Auswahl wird danach wiederhergestellt."""
    saved = layer.selectedFeatureIds()
    layer.selectByIds([feature.id()])
    try:
        return _render_site(canvas, layer, feature, float(cfg["radius_m"]), cfg["max_image_px"],
                            cfg["min_view_m"], cfg["draw_ring"])
    finally:
        layer.selectByIds(saved)


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


class _Out(QTextBrowser):  # pragma: no cover - braucht QGIS
    """Ergebnistext mit automatischem Zeilenumbruch und Scrollleiste - nichts wird mehr abgeschnitten."""

    def setText(self, html):
        self.setHtml(html)


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
        self.effort = QComboBox()
        self.effort.addItems(["(Standard)", "low", "medium", "high", "xhigh", "max"])
        self.effort.setCurrentIndex(EFFORTS.index(settings["effort"]))
        form.addRow("Effort (Denktiefe)", self.effort)
        self.thinking = QComboBox()
        self.thinking.addItems(["default (Modell entscheidet)", "off (kein / wenig Thinking)"])
        self.thinking.setCurrentIndex(1 if settings["thinking"] == "off" else 0)
        form.addRow("Thinking", self.thinking)
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
        self.prefetch = QSpinBox()
        self.prefetch.setRange(*_LIMITS["prefetch"][:2])
        self.prefetch.setValue(settings["prefetch"])
        self.prefetch.setToolTip("Nur im Auto-Modus: so viele Standorte vor dem aktuellen werden parallel schon "
                                 "eingeschätzt (0 = aus). Übersprungene/selbst entschiedene kosten trotzdem eine Anfrage.")
        form.addRow("Vorab einschätzen (Standorte)", self.prefetch)
        self.price_in = QDoubleSpinBox()
        self.price_in.setRange(0.0, 1000.0)
        self.price_in.setDecimals(3)
        self.price_in.setValue(settings["price_in"])
        self.price_out = QDoubleSpinBox()
        self.price_out.setRange(0.0, 1000.0)
        self.price_out.setDecimals(3)
        self.price_out.setValue(settings["price_out"])
        form.addRow("Preis Eingabe $/Mio. Token (0 = Tabelle)", self.price_in)
        form.addRow("Preis Ausgabe $/Mio. Token (0 = Tabelle)", self.price_out)
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
            "effort": EFFORTS[self.effort.currentIndex()],
            "thinking": "off" if self.thinking.currentIndex() == 1 else "default",
            "temperature": self.temp.value() if self.use_temp.isChecked() else None,
            "system_prompt": self.prompt.toPlainText(), "radius_m": self.radius.value(),
            "max_image_px": self.px.value(), "min_view_m": self.min_view.value(),
            "draw_ring": self.ring.isChecked(), "timeout_s": self.timeout.value(),
            "prefetch": self.prefetch.value(), "price_in": self.price_in.value(), "price_out": self.price_out.value(), "log_path": self.log_path.text()})


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
        self.out = _Out()
        self.out.setReadOnly(True)
        self.out.setMinimumHeight(130)
        self.out.setFrameShape(QFrame.Shape.NoFrame if hasattr(QFrame, "Shape") else QFrame.NoFrame)
        grid.addWidget(self.out, 5, 0, 1, 2)
        self.cost = QLabel("")
        self.cost.setStyleSheet("color: gray; font-size: 11px;")
        self.cost.setToolTip("Token dieses Checks, geschätzte Kosten und Summe seit dem Öffnen des Fensters "
                             "(inkl. vorab eingeschätzter Standorte). Preise: Tabelle im Skript, in den "
                             "Einstellungen überschreibbar.")
        grid.addWidget(self.cost, 6, 0, 1, 2)
        self.total = {"n": 0, "usd": 0.0, "unknown": False}
        note_btn = QPushButton("Als Notiz übernehmen")
        note_btn.clicked.connect(lambda _=False: self.to_note())
        grid.addWidget(note_btn, 8, 0, 1, 2)
        self.accept_btn = QPushButton("✔ Annehmen (ins gpkg)")
        self.deny_btn = QPushButton("✖ Ablehnen")
        self.accept_btn.clicked.connect(lambda _=False: self.decide_ai(True))
        self.deny_btn.clicked.connect(lambda _=False: self.decide_ai(False))
        for b in (self.accept_btn, self.deny_btn):
            b.setMinimumHeight(30)
            b.hide()
        grid.addWidget(self.accept_btn, 7, 0)
        grid.addWidget(self.deny_btn, 7, 1)
        self.pending = None                    # {"layer", "fid", "res"} bis angenommen/abgelehnt
        self.results = {}                      # fid -> geparstes Ergebnis (auch vorab eingeschätzte)
        self.errors = {}                       # fid -> Fehlertext (kein automatisches Wiederholen)
        self.inflight = set()                  # fids mit laufender Anfrage
        self.tasks = {}                        # Referenzen auf laufende QgsTasks
        self.shown = None                      # fid, dessen Ergebnis gerade angezeigt wird
        self.generation = 0                    # steigt bei Einstellungsänderung: alte Antworten verfallen
        self.auto = QCheckBox("Auto: nächsten Standort gleich einschätzen")
        self.auto.setChecked(self.settings["auto_next"])
        self.auto.toggled.connect(self._auto_toggled)
        self.model.currentTextChanged.connect(lambda _t: self._ui_changed())
        self.radius.valueChanged.connect(lambda _v: self._ui_changed())
        grid.addWidget(self.auto, 10, 0, 1, 2)
        self.timer = QTimer(self)              # prüft, ob im Review-Fenster ein neuer Standort dran ist
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._auto_tick)
        self.timer.start()
        gear = QPushButton("⚙ Einstellungen")
        gear.clicked.connect(lambda _=False: self.open_settings())
        grid.addWidget(gear, 9, 0, 1, 2)
        self.setWidget(body)
        iface.addDockWidget(_qt_enum("DockWidgetArea", "RightDockWidgetArea"), self)

    def _ui_changed(self):
        """Modell oder Umkreis im Fenster geändert: Einstellungen anpassen, alte Ergebnisse verwerfen."""
        self.settings["model"] = self.model.currentText().strip() or DEFAULT_MODEL
        self.settings["radius_m"] = float(self.radius.value())
        self.generation += 1
        self.results.clear()
        self.errors.clear()
        self.shown = None

    def _auto_toggled(self, on):
        self.settings["auto_next"] = bool(on)
        self._store()

    def _auto_tick(self):
        """Auto-Modus: zeigt das (ggf. schon vorab eingetroffene) Ergebnis des aktuellen Standorts und
        stößt Anfragen für den aktuellen sowie die nächsten `prefetch` Standorte der Warteschlange an
        (höchstens eine Bildberechnung pro Takt, die Anfragen selbst laufen parallel im Hintergrund).
        Nach 'Ablehnen' bleibt derselbe Standort dran: er wird nicht erneut eingeschätzt."""
        if not self.auto.isChecked():
            return
        panel = globals().get("_review_panel")
        try:
            todo = list(panel.queue.todo)
            if panel.awaiting_draw or not todo:
                return
        except Exception:
            return
        self._refresh_view()
        key = self._key_quiet()
        if not key:
            return
        n = self.settings["prefetch"]
        for fid in todo[:n + 1]:
            if fid in self.results or fid in self.errors or fid in self.inflight:
                continue
            if len(self.inflight) >= n + 1:
                break
            try:
                self._start(panel.layer, panel.feats[fid], key)
            except Exception as exc:
                self.errors[fid] = str(exc)
            self._refresh_view()
            break                                  # ein Render pro Takt, Oberfläche bleibt flüssig

    def _key_quiet(self):
        """API-Key ohne Dialog (Umgebung/Einstellungen); leer, wenn keiner da ist."""
        return (os.environ.get("ANTHROPIC_API_KEY", "").strip()
                or str(QgsSettings().value(SETTINGS_KEY, "") or "").strip())

    def _store(self):
        QgsSettings().setValue(PREFS_KEY, json.dumps(self.settings))

    def open_settings(self):
        self.settings["model"] = self.model.currentText().strip() or DEFAULT_MODEL
        self.settings["radius_m"] = float(self.radius.value())
        dlg = SettingsDialog(self, clean_settings(self.settings))
        if dlg.exec():
            self.settings = dlg.values()
            self.generation += 1               # alte/laufende Antworten gelten nicht mehr
            self.results.clear()
            self.errors.clear()
            self.shown = None
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

    def _site_info(self, fid=None):
        """site_id/Rang des Standorts (Review-Fenster), falls es läuft."""
        panel = globals().get("_review_panel")
        try:
            f = panel.feats[panel.queue.current if fid is None else fid]
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

    def _start(self, layer, feature, key, payload=None):
        """Bild rendern (Hauptthread) und die Anfrage im Hintergrund losschicken; das Ergebnis landet
        in self.results[fid]. payload: fertige Anfrage (Fallback: Kartenausschnitt statt Standort)."""
        cfg = self.settings
        fid = feature.id() if feature is not None else "canvas"
        if payload is None:
            image = _render_selected(self.iface.mapCanvas(), layer, feature, cfg) \
                if cfg["radius_m"] > 0 else self.iface.mapCanvas().grab().toImage()
            payload = build_payload(base64.b64encode(_image_png(image, cfg["max_image_px"])).decode("ascii"),
                                    "image/png", self.context.text(), cfg["model"], cfg["max_tokens"],
                                    cfg["system_prompt"], cfg["temperature"], cfg["effort"], cfg["thinking"])
        info = {"site_id": "", "rank": ""} if fid == "canvas" else self._site_info(fid)
        gen = self.generation
        self.inflight.add(fid)

        def work(task):
            return run_check(payload, key, timeout=cfg["timeout_s"])

        def done(exception, result=None):
            self.inflight.discard(fid)
            self.tasks.pop(fid, None)
            if gen != self.generation:         # Einstellungen inzwischen geändert: Antwort verwerfen
                return
            if exception is not None or result is None:
                self.errors[fid] = str(exception)
            else:
                res = parse_reply(result)
                self.results[fid] = res
                self._add_total(res)
                try:
                    log_row(cfg["log_path"], {**info, "model": payload["model"], "effort": cfg["effort"],
                                              "thinking": cfg["thinking"], "prompt_id": prompt_id(cfg["system_prompt"]),
                                              "label": res["label"], "confidence": res["confidence"],
                                              "dump_probability": res["dump_probability"],
                                              "dump_location": res["dump_location"], "reason": res["reason"]})
                except OSError:
                    pass
            if fid == "canvas":
                self._show("canvas", None)
            else:
                self._refresh_view()

        task = QgsTask.fromFunction("Claude Einschätzung", work, on_finished=done)
        self.tasks[fid] = task
        QgsApplication.taskManager().addTask(task)

    def _refresh_view(self):
        """Zeigt zum aktuellen Standort das Ergebnis (falls da), den Fehler oder 'Frage Claude ...'."""
        layer, feature = self._target()
        if feature is None:
            return
        fid = feature.id()
        if fid == self.shown:
            return
        if fid in self.results or fid in self.errors:
            self._show(fid, layer)
        else:
            self.pending = None
            for b in (self.accept_btn, self.deny_btn, self.banner):
                b.hide()
            self.out.setText("Frage Claude ..." if fid in self.inflight else "")

    def _cost_of(self, res):
        cfg = self.settings
        return estimate_cost(cfg["model"], res["tokens_in"], res["tokens_out"], cfg["price_in"], cfg["price_out"])

    def _add_total(self, res):
        usd = self._cost_of(res)
        self.total["n"] += 1
        if usd is None:
            self.total["unknown"] = True
        else:
            self.total["usd"] += usd

    def _cost_text(self, res=None):
        tot = self.total
        line = f"Σ {tot['n']} Checks " + ("(Preis unbekannt)" if tot["unknown"] and not tot["usd"]
                                          else fmt_cost(tot["usd"]))
        if res is None:
            return line
        return (f"{res['tokens_in']:,} ein / {res['tokens_out']:,} aus  {fmt_cost(self._cost_of(res))}\n" + line)

    def _show(self, fid, layer):
        """Ergebnis (oder Fehler) von fid in Kasten und Text anzeigen; bei lesbarem Ergebnis auch
        Annehmen/Ablehnen anbieten."""
        self.shown = fid
        self.pending = None
        self.accept_btn.hide()
        self.deny_btn.hide()
        res = self.results.get(fid)
        self.cost.setText(self._cost_text(res))
        if res is None:
            self.banner.hide()
            self.out.setText(f"Fehler: {self.errors.get(fid, '?')}")
            return
        self.last_result = res
        text, color = banner(res)
        self.banner.setText(text)
        self.banner.setStyleSheet(f"background: {color}; color: white; font-size: 20px; "
                                  "font-weight: bold; border-radius: 6px; padding: 6px;")
        self.banner.show()
        if not res["ok"]:
            self.out.setText(f"Antwort nicht lesbar (stop_reason: {res['stop_reason'] or '?'}). "
                             "Bei max_tokens: Limit in den Einstellungen erhöhen oder Thinking aus / "
                             f"Effort niedriger.<br>{res['reason'] or '(leer)'}")
            return
        if layer is not None:
            self.pending = {"layer": layer, "fid": fid, "res": res}
            self.accept_btn.setEnabled(True)
            self.deny_btn.setEnabled(True)
            self.accept_btn.show()
            self.deny_btn.show()
        self.out.setText(f"<b>{res['label']}</b> &nbsp; ({res['confidence']:.2f})"
                         f"<br>Halde: {res['dump_location'] or '-'}<br>{res['reason']}")

    def classify(self):
        """Knopf 'Einschätzen': aktuellen Standort (neu) einschätzen, auch wenn schon ein Ergebnis da ist."""
        key = self._key()
        if not key:
            self.out.setText("Kein API-Key.")
            return
        self.settings = clean_settings({**self.settings, "model": self.model.currentText(),
                                        "radius_m": self.radius.value()})
        self._store()
        layer, feature = self._target()
        fid = feature.id() if feature is not None else "canvas"
        if fid in self.inflight:
            return
        self.results.pop(fid, None)
        self.errors.pop(fid, None)
        self.shown = None
        self.pending = None
        for b in (self.accept_btn, self.deny_btn, self.banner):
            b.hide()
        self.out.setText("Frage Claude ...")
        try:
            if feature is None:                # kein Standort gewählt: sichtbaren Ausschnitt nehmen
                canvas = self.iface.mapCanvas()
                cfg = self.settings
                png = _image_png(canvas.grab().toImage(), cfg["max_image_px"])
                payload = build_payload(base64.b64encode(png).decode("ascii"), "image/png", self.context.text(),
                                        cfg["model"], cfg["max_tokens"], cfg["system_prompt"], cfg["temperature"],
                                        cfg["effort"], cfg["thinking"])
                self._start(None, None, key, payload)
            else:
                self._start(layer, feature, key)
        except Exception as exc:
            self.errors[fid] = str(exc)
            self._show(fid, layer)

    def decide_ai(self, accepted):
        """Annehmen: KI-Urteil als review in die Schicht (über das Review-Fenster, falls offen);
        Ablehnen: nur vermerken, review bleibt leer. Beides wird in ai_* Spalten gespeichert."""
        pend = self.pending
        if pend is None:
            return
        layer, fid, res = pend["layer"], pend["fid"], pend["res"]
        try:
            _ensure_ai_fields(layer)
            fields = layer.fields()
            for name, value in (("ai_label", res["label"]), ("ai_dump_p", round(res["dump_probability"], 3)),
                                ("ai_decision", "accepted" if accepted else "denied")):
                layer.changeAttributeValue(fid, fields.indexOf(name), value)
            layer.commitChanges(False)
            if not layer.isEditable():
                layer.startEditing()
        except Exception as exc:
            self.out.setText(f"Speichern fehlgeschlagen: {exc}")
            return
        value = verdict_to_review(res)
        panel = globals().get("_review_panel")
        in_queue = False
        if accepted:
            try:
                in_queue = panel is not None and panel.queue.current == fid
            except Exception:
                in_queue = False
            note = ai_note(res)
            if in_queue:
                panel.note.setEditText(note)
                panel.decide(value)            # trägt ein, springt weiter, sichert regelmäßig
            else:
                fields = layer.fields()
                layer.changeAttributeValue(fid, fields.indexOf("review"), value)
                if fields.indexOf("note") >= 0:
                    layer.changeAttributeValue(fid, fields.indexOf("note"), note)
                layer.commitChanges(False)
                if not layer.isEditable():
                    layer.startEditing()
        if panel is not None:
            try:
                panel.save_to_file(quiet=True)
            except Exception:
                pass
        self.pending = None
        self.accept_btn.hide()
        self.deny_btn.hide()
        self.banner.hide()
        self.out.setText(f"Angenommen: review = {value}" if accepted else "Abgelehnt (review bleibt leer)")

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
