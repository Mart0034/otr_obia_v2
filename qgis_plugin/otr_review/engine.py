"""Anfragen an die KI: Bild rendern, im Hintergrund senden, Ergebnis melden."""

import base64
import time

from qgis.core import QgsApplication, QgsTask

from .core import (build_payload, log_row, parse_reply, prompt_id, radius_for, run_check, zone_note)
from .render import image_png, render_selected, render_view_clean, start_render_clean


def feature_context(feature):
    """Wert der Spalte `context` (inside_op / near_op / open_desert) oder ''."""
    return str(feature["context"]) if feature.fields().indexOf("context") >= 0 and feature["context"] else ""


class AiEngine:
    """submit() rendert den Standort und schickt die Anfrage im Hintergrund los; callback(fid, res, error,
    meta) läuft danach im Hauptthread. meta: radius_m, context, model, render_s, api_s (Sekunden).
    render_mode 'clean' rendert ebenfalls im Hintergrund (mehrere Standorte gleichzeitig, `rendering` zählt
    sie); 'visible' blockiert QGIS kurz je Standort."""

    def __init__(self, iface, get_settings, get_extra_context, get_site_info, get_examples=None):
        self.iface = iface
        self.get_settings = get_settings          # -> aktuelle Einstellungen (dict)
        self.get_extra_context = get_extra_context  # -> Zusatzinfo-Text des Fensters
        self.get_site_info = get_site_info        # fid -> {"site_id", "rank"}
        self.get_examples = get_examples or (lambda cfg: [])   # cfg -> Beispielbilder für die Anfrage
        self.inflight = set()                     # Standorte in Arbeit (Rendern + Anfrage)
        self.rendering = 0                        # davon gerade beim Rendern
        self.tasks = {}
        self.generation = 0                       # steigt bei Einstellungsänderung: alte Antworten verfallen

    def invalidate(self):
        self.generation += 1

    def submit(self, layer, feature, key, callback):
        cfg = dict(self.get_settings())
        fid = feature.id()
        context = feature_context(feature)
        radius = radius_for(context, cfg)
        gen = self.generation
        meta = {"radius_m": radius, "context": context, "model": cfg["model"], "render_s": 0.0, "api_s": 0.0}
        t0 = time.time()
        self.inflight.add(fid)

        def fail(exc):
            self.inflight.discard(fid)
            if gen == self.generation:
                callback(fid, None, str(exc), meta)

        def send(image, m_per_px=None):
            try:
                meta["render_s"] = time.time() - t0
                meta["m_per_px"] = m_per_px
                extra = "; ".join(t for t in (zone_note(context, radius, cfg["tell_context"], m_per_px),
                                              self.get_extra_context()) if t)
                payload = build_payload(base64.b64encode(image_png(image, cfg["max_image_px"])).decode("ascii"),
                                        "image/png", extra, cfg["model"], cfg["max_tokens"], cfg["system_prompt"],
                                        cfg["temperature"], cfg["effort"], cfg["thinking"], cfg["prompt_cache"],
                                        self.get_examples(cfg) if cfg.get("use_examples") else None)
                self._send(fid, payload, key, cfg, gen, meta, callback)
            except Exception as exc:
                fail(exc)

        canvas = self.iface.mapCanvas()
        try:
            if radius > 0 and cfg["render_mode"] == "clean":
                self.rendering += 1

                def rendered(image, m_per_px):
                    self.rendering -= 1
                    if image is None:               # kein sichtbares Satellitenbild: wie 'visible'
                        image, m_per_px = render_selected(canvas, layer, feature, cfg, radius)
                    send(image, m_per_px)

                start_render_clean(canvas, layer, feature, cfg, radius, rendered)
            else:
                if radius > 0:
                    send(*render_selected(canvas, layer, feature, cfg, radius))
                else:
                    send(canvas.grab().toImage(), None)
        except Exception as exc:
            fail(exc)

    def render_image(self, layer, feature, callback):
        """Bild eines Standorts genau so rendern, wie die KI es sieht (für 'Als Beispiel speichern');
        callback(image, m_per_px) im Hauptthread."""
        cfg = dict(self.get_settings())
        radius = radius_for(feature_context(feature), cfg)
        canvas = self.iface.mapCanvas()
        if radius > 0 and cfg["render_mode"] == "clean":
            def rendered(image, m_per_px):
                if image is None:
                    image, m_per_px = render_selected(canvas, layer, feature, cfg, radius)
                callback(image, m_per_px)
            start_render_clean(canvas, layer, feature, cfg, radius, rendered)
        elif radius > 0:
            callback(*render_selected(canvas, layer, feature, cfg, radius))
        else:
            callback(canvas.grab().toImage(), None)

    def render_view(self, callback):
        """Den sichtbaren Kartenausschnitt wie für die KI rendern (Beispielbild von einer beliebigen Stelle)."""
        cfg = dict(self.get_settings())
        image, m_per_px = render_view_clean(self.iface.mapCanvas(), cfg["example_px"])
        if image is None:
            image, m_per_px = self.iface.mapCanvas().grab().toImage(), None
        callback(image, m_per_px)

    def _send(self, fid, payload, key, cfg, gen, meta, callback):
        info = self.get_site_info(fid)
        t1 = time.time()

        def work(task):
            return run_check(payload, key, timeout=cfg["timeout_s"])

        def done(exception, result=None):
            self.inflight.discard(fid)
            self.tasks.pop(fid, None)
            if gen != self.generation:             # Einstellungen inzwischen geändert: Antwort verwerfen
                return
            meta["api_s"] = time.time() - t1
            if exception is not None or result is None:
                callback(fid, None, str(exception), meta)
                return
            res = parse_reply(result)
            res["radius_m"], res["context"] = meta["radius_m"], meta["context"]
            try:
                log_row(cfg["log_path"], {**info, "model": cfg["model"], "effort": cfg["effort"],
                                          "thinking": cfg["thinking"], "prompt_id": prompt_id(cfg["system_prompt"]),
                                          "label": res["label"], "confidence": res["confidence"],
                                          "dump_probability": res["dump_probability"],
                                          "dump_location": res["dump_location"], "reason": res["reason"],
                                          "radius_m": meta["radius_m"], "context": meta["context"],
                                          "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"]})
            except OSError:
                pass
            callback(fid, res, None, meta)

        task = QgsTask.fromFunction("Claude Einschätzung", work, on_finished=done)
        self.tasks[fid] = task
        QgsApplication.taskManager().addTask(task)
