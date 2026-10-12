"""Beispielbilder: von dir bestätigte Referenzen (echte Reifenhalde / keine Halde, z.B. Erdhaufen), die mit
jeder KI-Anfrage vor dem neuen Bild mitgeschickt werden. Ohne Qt, nur Dateien: je Beispiel eine PNG und eine
JSON-Datei (kind, caption, site_id, enabled) im Ordner."""

import base64
import json
import os
import time

from .core import pick_examples

DEFAULT_CAPTIONS = {
    "dump": "tires in rows, stacks and piles (hollow rings, 2.5-4 m)",
    "not_dump": "dirt / rock heaps and shadows - soft mounds, not tires",
}


class ExampleStore:
    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self._cache = {}                 # id -> b64 (PNG-Dateien nur einmal lesen)

    def _meta_path(self, ex_id):
        return os.path.join(self.folder, f"{ex_id}.json")

    def _png_path(self, ex_id):
        return os.path.join(self.folder, f"{ex_id}.png")

    def add(self, png_bytes, kind, caption="", site_id=""):
        """Beispiel speichern; gibt die id zurück. kind: 'dump' oder 'not_dump'."""
        if kind not in DEFAULT_CAPTIONS:
            raise ValueError(f"kind muss dump oder not_dump sein, nicht {kind!r}")
        ex_id = f"{int(time.time() * 1000)}"
        while os.path.exists(self._meta_path(ex_id)):
            ex_id = str(int(ex_id) + 1)
        with open(self._png_path(ex_id), "wb") as fh:
            fh.write(png_bytes)
        meta = {"id": ex_id, "kind": kind, "caption": caption.strip() or DEFAULT_CAPTIONS[kind],
                "site_id": site_id, "enabled": True}
        with open(self._meta_path(ex_id), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        return ex_id

    def list(self):
        """Alle Beispiele (dicts ohne Bilddaten), älteste zuerst."""
        out = []
        for name in sorted(os.listdir(self.folder)):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(self.folder, name), encoding="utf-8") as fh:
                        meta = json.load(fh)
                except (OSError, ValueError):
                    continue
                if meta.get("kind") in DEFAULT_CAPTIONS and os.path.exists(self._png_path(meta.get("id", ""))):
                    out.append(meta)
        return sorted(out, key=lambda m: m["id"])

    def set_enabled(self, ex_id, enabled):
        meta = next((m for m in self.list() if m["id"] == ex_id), None)
        if meta is None:
            return False
        meta["enabled"] = bool(enabled)
        with open(self._meta_path(ex_id), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        return True

    def set_caption(self, ex_id, caption):
        meta = next((m for m in self.list() if m["id"] == ex_id), None)
        if meta is None:
            return False
        meta["caption"] = caption.strip() or DEFAULT_CAPTIONS[meta["kind"]]
        with open(self._meta_path(ex_id), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        return True

    def delete(self, ex_id):
        for path in (self._meta_path(ex_id), self._png_path(ex_id)):
            if os.path.exists(path):
                os.remove(path)
        self._cache.pop(ex_id, None)

    def png_bytes(self, ex_id):
        with open(self._png_path(ex_id), "rb") as fh:
            return fh.read()

    def active(self, max_n):
        """Die für die nächste Anfrage zu verwendenden Beispiele (aktiv, ausgewogen, höchstens max_n) samt
        Bild als base64 - Reihenfolge stabil, damit der Prompt-Cache greift."""
        chosen = pick_examples([m for m in self.list() if m.get("enabled", True)], max_n)
        out = []
        for m in chosen:
            if m["id"] not in self._cache:
                self._cache[m["id"]] = base64.b64encode(self.png_bytes(m["id"])).decode("ascii")
            out.append({**m, "b64": self._cache[m["id"]]})
        return out
