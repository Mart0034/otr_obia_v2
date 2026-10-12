"""Anfragen an die KI: Bild rendern (Hauptthread), im Hintergrund senden, Ergebnis melden."""

import base64

from qgis.core import QgsApplication, QgsTask

from .core import (build_payload, log_row, parse_reply, prompt_id, radius_for, run_check, zone_note)
from .render import image_png, render_selected


def feature_context(feature):
    """Wert der Spalte `context` (inside_op / near_op / open_desert) oder ''."""
    return str(feature["context"]) if feature.fields().indexOf("context") >= 0 and feature["context"] else ""


class AiEngine:
    """submit() rendert den Standort sofort und schickt die Anfrage im Hintergrund los; callback(fid,
    res, error, meta) läuft danach im Hauptthread. meta: radius_m, context, model."""

    def __init__(self, iface, get_settings, get_extra_context, get_site_info):
        self.iface = iface
        self.get_settings = get_settings          # -> aktuelle Einstellungen (dict)
        self.get_extra_context = get_extra_context  # -> Zusatzinfo-Text des Fensters
        self.get_site_info = get_site_info        # fid -> {"site_id", "rank"}
        self.inflight = set()
        self.tasks = {}
        self.generation = 0                       # steigt bei Einstellungsänderung: alte Antworten verfallen

    def invalidate(self):
        self.generation += 1

    def submit(self, layer, feature, key, callback):
        cfg = dict(self.get_settings())
        fid = feature.id()
        context = feature_context(feature)
        radius = radius_for(context, cfg)
        canvas = self.iface.mapCanvas()
        image = render_selected(canvas, layer, feature, cfg, radius) if radius > 0 else canvas.grab().toImage()
        extra = "; ".join(t for t in (zone_note(context, radius), self.get_extra_context()) if t)
        payload = build_payload(base64.b64encode(image_png(image, cfg["max_image_px"])).decode("ascii"),
                                "image/png", extra, cfg["model"], cfg["max_tokens"], cfg["system_prompt"],
                                cfg["temperature"], cfg["effort"], cfg["thinking"], cfg["prompt_cache"])
        info = self.get_site_info(fid)
        gen = self.generation
        meta = {"radius_m": radius, "context": context, "model": cfg["model"]}
        self.inflight.add(fid)

        def work(task):
            return run_check(payload, key, timeout=cfg["timeout_s"])

        def done(exception, result=None):
            self.inflight.discard(fid)
            self.tasks.pop(fid, None)
            if gen != self.generation:             # Einstellungen inzwischen geändert: Antwort verwerfen
                return
            if exception is not None or result is None:
                callback(fid, None, str(exception), meta)
                return
            res = parse_reply(result)
            res["radius_m"], res["context"] = radius, context
            try:
                log_row(cfg["log_path"], {**info, "model": cfg["model"], "effort": cfg["effort"],
                                          "thinking": cfg["thinking"], "prompt_id": prompt_id(cfg["system_prompt"]),
                                          "label": res["label"], "confidence": res["confidence"],
                                          "dump_probability": res["dump_probability"],
                                          "dump_location": res["dump_location"], "reason": res["reason"],
                                          "radius_m": radius, "context": context,
                                          "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"]})
            except OSError:
                pass
            callback(fid, res, None, meta)

        task = QgsTask.fromFunction("Claude Einschätzung", work, on_finished=done)
        self.tasks[fid] = task
        QgsApplication.taskManager().addTask(task)
