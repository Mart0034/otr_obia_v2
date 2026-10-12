"""Das eine Fenster des Plugins: Review-Warteschlange und KI-Einschätzung zusammen."""

import json
import os

from qgis.core import QgsCoordinateTransform, QgsProject, QgsRectangle, QgsSettings

from .batch import BatchRunner
from .core import (DEFAULT_MODEL, ai_overview, fmt_overview, KEY_SETTING, OLD_PREFS_KEY, PREFS_KEY, ReviewQueue, ai_note, audit_summary,
                   banner, clean_settings, cost_of, fmt_cost, human_followup, load_settings, result_from_fields,
                   verdict_to_review)
from .engine import AiEngine, feature_context
from .layers import ensure_ai_fields, layer_by_name
from .qtcompat import (QCheckBox, QComboBox, QDesktopServices, QDockWidget, QFrame, QGridLayout, QGroupBox,
                       QInputDialog, QKeySequence, QLabel, QLineEdit, QProgressBar, QPushButton,
                       QScrollArea, QShortcut, QSpinBox, QTextBrowser, QTimer, QUrl, QVBoxLayout, QWidget, enum,
                       exec_dialog, qt)
from .settings_dialog import SettingsDialog

MIN_VIEW_M = 400.0   # kleinster Kartenausschnitt (Meter) beim Heranzoomen
NOTE_PRESETS = ["", "town", "yard / equipment", "road / haul road", "pit / rock pile", "neat dump", "messy dump",
                "rubber, unclear"]
DECISIONS = (("1  Dump", "dump"), ("2  Clean", "clean"), ("3  Partial", "partial"), ("4  Unsure", "unsure"))


class _Out(QTextBrowser):
    """Ergebnistext mit Zeilenumbruch und Scrollleiste - nichts wird abgeschnitten."""

    def setText(self, html):
        self.setHtml(html)


class OtrPanel(QDockWidget):
    def __init__(self, iface, layer):
        super().__init__("OTR Review", iface.mainWindow())
        self.setObjectName("OtrReviewPanel")
        self.iface = iface
        self.layer = layer
        self.drawn = layer_by_name("drawn_polygons")
        self.setFocusPolicy(qt("FocusPolicy", "StrongFocus"))
        self.awaiting_draw = False
        self.since_save = 0
        self.batch = None

        # ---- Einstellungen / KI-Zustand
        self.settings = self._load_settings()
        self.results = {}          # fid -> Ergebnis dieser Sitzung (oder aus der Datei geladen)
        self.errors = {}           # fid -> Fehlertext (kein automatisches Wiederholen)
        self.shown = None          # fid, dessen Ergebnis gerade angezeigt wird
        self.pending = None        # {"fid", "res"}: Ergebnis wartet auf Annehmen/Ablehnen
        self.total = {"n": 0, "usd": 0.0, "unknown": False}
        self.engine = AiEngine(iface, lambda: self.settings, lambda: self.context.text(), self._site_info)

        # ---- Schicht vorbereiten (Spalten, Bearbeitungsmodus), Warteschlange
        ensure_ai_fields(layer)
        self.idx = layer.fields().indexOf("review")
        self.note_idx = layer.fields().indexOf("note")
        self.feats = {f.id(): f for f in layer.getFeatures()}
        self.queue = ReviewQueue([(fid, f["rank"], f["review"] if f["review"] else "",
                                   f["note"] if f["note"] else "") for fid, f in self.feats.items()])
        self._build_ui()
        for key, func in (("1", lambda: self.decide("dump")), ("2", lambda: self.decide("clean")),
                          ("3", lambda: self.decide("partial")), ("4", lambda: self.decide("unsure")),
                          ("5", self.skip), ("6", self.back), ("7", self.open_map)):
            QShortcut(QKeySequence(key), self, activated=func, context=qt("ShortcutContext", "ApplicationShortcut"))
        iface.addDockWidget(qt("DockWidgetArea", "RightDockWidgetArea"), self)
        if not layer.isEditable():
            layer.startEditing()
        self.timer = QTimer(self)             # Auto-Modus: prüft, ob ein neuer Standort dran ist
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._auto_tick)
        self.timer.start()
        self.show_current()
        self._refresh_stats()

    # =====================================================================================
    # Aufbau
    # =====================================================================================
    def _build_ui(self):
        body = QWidget()
        col = QVBoxLayout(body)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        col.addWidget(self.info)
        self.last = QLabel("")
        self.last.setWordWrap(True)
        self.last.setStyleSheet("font-weight: bold; color: #1a7f37;")
        col.addWidget(self.last)

        back_btn = QPushButton("6  ◀ Zurück  (letzte Entscheidung rückgängig)")
        back_btn.setMinimumHeight(30)
        back_btn.clicked.connect(lambda _=False: self.back())
        col.addWidget(back_btn)
        self.note = QComboBox()
        self.note.setEditable(True)
        self.note.addItems(NOTE_PRESETS)
        self.note.setCurrentIndex(0)
        self.note.lineEdit().setPlaceholderText("Notiz (z.B. town, yard, road) - dann Entscheidung wählen")
        self.note.activated.connect(lambda _=0: self.setFocus())
        self.note.lineEdit().returnPressed.connect(self.setFocus)
        col.addWidget(self.note)
        grid = QGridLayout()
        for n, (text, value) in enumerate(DECISIONS):
            btn = QPushButton(text)
            btn.setMinimumHeight(28)
            btn.clicked.connect(lambda _=False, v=value: self.decide(v))
            grid.addWidget(btn, n // 2, n % 2)
        extra = [("5  Skip", self.skip), ("7  Karte", self.open_map), ("Teil einzeichnen", self.draw_part),
                 ("Weiter", self.next_after_draw), ("💾 Sichern", self.save_to_file)]
        for n, (text, func) in enumerate(extra):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, f=func: f())
            grid.addWidget(btn, 2 + n // 2, n % 2)
        col.addLayout(grid)

        # ---- KI
        box = QGroupBox("KI-Einschätzung")
        ai = QGridLayout(box)
        self.model = QComboBox()
        self.model.setEditable(True)
        from .core import MODELS
        self.model.addItems(MODELS)
        self.model.setEditText(self.settings["model"])
        ai.addWidget(QLabel("Modell"), 0, 0)
        ai.addWidget(self.model, 0, 1)
        self.context = QLineEdit()
        self.context.setPlaceholderText("Zusatzinfo für die KI (optional, z.B. 'Satellitenbild 0.5 m')")
        ai.addWidget(self.context, 1, 0, 1, 2)
        self.radius = QSpinBox()
        self.radius.setRange(0, 500)
        self.radius.setValue(int(self.settings["radius_m"]))
        self.radius.setSuffix(" m Umkreis (ohne Umgebungsspalte)")
        ai.addWidget(self.radius, 2, 0, 1, 2)
        self.go = QPushButton("Einschätzen (Standort + Umkreis)")
        self.go.setMinimumHeight(30)
        self.go.clicked.connect(lambda _=False: self.classify())
        ai.addWidget(self.go, 3, 0, 1, 2)
        self.banner = QLabel("")
        self.banner.setAlignment(qt("AlignmentFlag", "AlignCenter"))
        self.banner.setMinimumHeight(48)
        self.banner.hide()
        ai.addWidget(self.banner, 4, 0, 1, 2)
        self.out = _Out()
        self.out.setReadOnly(True)
        self.out.setMinimumHeight(120)
        self.out.setFrameShape(enum(QFrame, "Shape", "NoFrame"))
        ai.addWidget(self.out, 5, 0, 1, 2)
        self.cost = QLabel("")
        self.cost.setStyleSheet("color: gray; font-size: 11px;")
        ai.addWidget(self.cost, 6, 0, 1, 2)
        self.accept_btn = QPushButton("✔ Annehmen (ins gpkg)")
        self.deny_btn = QPushButton("✖ Ablehnen")
        self.accept_btn.clicked.connect(lambda _=False: self.decide_ai(True))
        self.deny_btn.clicked.connect(lambda _=False: self.decide_ai(False))
        for b in (self.accept_btn, self.deny_btn):
            b.setMinimumHeight(28)
            b.hide()
        ai.addWidget(self.accept_btn, 7, 0)
        ai.addWidget(self.deny_btn, 7, 1)
        self.auto = QCheckBox("Auto: aktuellen + nächste Standorte gleich einschätzen")
        self.auto.setChecked(self.settings["auto_next"])
        self.auto.toggled.connect(self._auto_toggled)
        ai.addWidget(self.auto, 8, 0, 1, 2)
        col.addWidget(box)

        # ---- Stapel
        bbox = QGroupBox("Stapel-Analyse")
        bl = QGridLayout(bbox)
        self.batch_n = QSpinBox()
        self.batch_n.setRange(1, 100000)
        self.batch_n.setValue(self.settings["batch_n"])
        self.batch_n.setSuffix(" Standorte")
        bl.addWidget(self.batch_n, 0, 0, 1, 2)
        self.batch_btn = QPushButton("▶ Stapel starten")
        self.batch_btn.clicked.connect(lambda _=False: self.start_batch("batch"))
        self.test_btn = QPushButton("Test auf Auswahl (ändert nichts)")
        self.test_btn.clicked.connect(lambda _=False: self.start_batch("test"))
        self.cancel_btn = QPushButton("■ Abbrechen")
        self.cancel_btn.clicked.connect(lambda _=False: self.cancel_batch())
        self.cancel_btn.hide()
        bl.addWidget(self.batch_btn, 1, 0)
        bl.addWidget(self.test_btn, 1, 1)
        bl.addWidget(self.cancel_btn, 2, 0, 1, 2)
        self.progress = QProgressBar()
        self.progress.hide()
        bl.addWidget(self.progress, 3, 0, 1, 2)
        self.batch_info = QLabel("")
        self.batch_info.setWordWrap(True)
        bl.addWidget(self.batch_info, 4, 0, 1, 2)
        self.overview = QLabel("")
        self.overview.setWordWrap(True)
        self.overview.setStyleSheet("font-weight: bold;")
        bl.addWidget(self.overview, 5, 0, 1, 2)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        self.stats.setStyleSheet("color: gray; font-size: 11px;")
        bl.addWidget(self.stats, 6, 0, 1, 2)
        col.addWidget(bbox)

        gear = QPushButton("⚙ Einstellungen")
        gear.clicked.connect(lambda _=False: self.open_settings())
        col.addWidget(gear)
        col.addStretch(1)
        self.model.currentTextChanged.connect(lambda _t: self._ui_changed())
        self.radius.valueChanged.connect(lambda _v: self._ui_changed())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        scroll.setFrameShape(enum(QFrame, "Shape", "NoFrame"))
        self.setWidget(scroll)

    # =====================================================================================
    # Einstellungen
    # =====================================================================================
    def _load_settings(self):
        qs = QgsSettings()
        text = qs.value(PREFS_KEY, "") or qs.value(OLD_PREFS_KEY, "")
        return load_settings(text)

    def _store(self):
        QgsSettings().setValue(PREFS_KEY, json.dumps(self.settings))

    def store_settings(self, new):
        self.settings = new
        self._store()
        self.engine.invalidate()
        self.results.clear()
        self.errors.clear()
        self.shown = None
        self.model.setEditText(new["model"])
        self.radius.setValue(int(new["radius_m"]))

    def open_settings(self):
        self.settings["model"] = self.model.currentText().strip() or DEFAULT_MODEL
        self.settings["radius_m"] = float(self.radius.value())
        dlg = SettingsDialog(self, clean_settings(self.settings))
        if exec_dialog(dlg):
            self.store_settings(clean_settings({**self.settings, **dlg.values()}))   # auto_next, batch_n bleiben
            self.out.setText("Einstellungen gespeichert.")

    def _ui_changed(self):
        """Modell oder Umkreis im Fenster geändert: Einstellungen anpassen, alte Ergebnisse verwerfen."""
        self.settings["model"] = self.model.currentText().strip() or DEFAULT_MODEL
        self.settings["radius_m"] = float(self.radius.value())
        self.engine.invalidate()
        self.results.clear()
        self.errors.clear()
        self.shown = None

    def _auto_toggled(self, on):
        self.settings["auto_next"] = bool(on)
        self._store()

    def _key(self, ask=True):
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        qs = QgsSettings()
        if not key:
            key = str(qs.value(KEY_SETTING, "") or "").strip()
        if not key and ask:
            echo = enum(QLineEdit, "EchoMode", "Password")
            key, ok = QInputDialog.getText(self, "Anthropic API-Key", "API-Key (sk-ant-...):", echo)
            key = key.strip() if ok else ""
            if key:
                qs.setValue(KEY_SETTING, key)
        return key

    # =====================================================================================
    # Review-Teil (wie früher review_queue.py)
    # =====================================================================================
    def _site_info(self, fid=None):
        try:
            f = self.feats[self.queue.current if fid is None else fid]
            return {"site_id": f["site_id"], "rank": f["rank"]}
        except Exception:
            return {"site_id": "", "rank": ""}

    def show_current(self):
        fid = self.queue.current
        counts = ", ".join(f"{k} {v}" for k, v in sorted(self.queue.counts().items())) or "noch nichts"
        if fid is None:
            self.layer.removeSelection()
            self.info.setText(f"Fertig. Alle {self.queue.total} Standorte haben ein Urteil.\n{counts}")
            return
        f = self.feats[fid]
        ctx = feature_context(f)
        self.info.setText(
            f"<b>{f['site_id']}</b> &nbsp; Rang {f['rank']}<br>{f['mine_id']} &nbsp; {f['size_class']}"
            + (f" &nbsp; <i>{ctx}</i>" if ctx else "") + "<br>"
            f"{round(f['area_m2']):,} m² &nbsp; Score {f['mean_proba']:.2f}"
            + ("<br><b>nahe bekannter Halde</b>" if f.fields().indexOf("near_known_dump") >= 0 and f["near_known_dump"] else "")
            + f"<br><br>{self.queue.done} von {self.queue.total} erledigt<br><small>{counts}</small>")
        self.layer.selectByIds([fid])
        box = f.geometry().boundingBox()
        box.scale(2.0)
        if box.width() < MIN_VIEW_M or box.height() < MIN_VIEW_M:
            c = box.center()
            box = QgsRectangle(c.x() - MIN_VIEW_M / 2, c.y() - MIN_VIEW_M / 2,
                               c.x() + MIN_VIEW_M / 2, c.y() + MIN_VIEW_M / 2)
        canvas = self.iface.mapCanvas()
        xform = QgsCoordinateTransform(self.layer.crs(), canvas.mapSettings().destinationCrs(), QgsProject.instance())
        canvas.setExtent(xform.transformBoundingBox(box))
        canvas.refresh()

    def notify(self, text):
        self.last.setText(text)
        try:
            bar = self.iface.messageBar()
            bar.clearWidgets()
            bar.pushSuccess("OTR Review", text)
        except Exception:
            pass

    def _write(self, fid, value, note=None):
        self.layer.changeAttributeValue(fid, self.idx, value if value else None)
        if note is not None and self.note_idx >= 0:
            self.layer.changeAttributeValue(fid, self.note_idx, note if note else None)
        self._commit()

    def _commit(self):
        self.layer.commitChanges(False)           # sofort speichern, Bearbeitungsmodus bleibt an
        if not self.layer.isEditable():
            self.layer.startEditing()

    def save_to_file(self, quiet=False):
        """Schreibt die in der Seitendatei (-wal) wartenden Änderungen in die eigentliche .gpkg-Datei."""
        import sqlite3
        path = self.layer.source().split("|")[0]
        try:
            con = sqlite3.connect(path, timeout=10)
            busy, _, _ = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            con.close()
            if not quiet:
                self.notify(f"In die Datei gesichert: {path}" + (" (teilweise - QGIS liest gerade)" if busy else ""))
        except Exception as exc:
            self.notify(f"Sichern fehlgeschlagen: {exc}")

    def decide(self, value):
        if self.awaiting_draw:
            self.next_after_draw()
        note = self.note.currentText().strip()
        fid = self.queue.decide(value, note if note else None)
        if fid is not None:
            self._write(fid, value, note if note else None)
            self._followup(fid, value)
            f = self.feats[fid]
            self.notify(f"Standort {f['site_id']} (Rang {f['rank']}) als {value.upper()} eingetragen"
                        + (f" - Notiz: {note}" if note else ""))
        self.note.setCurrentIndex(0)
        self.note.clearEditText()
        self.since_save += 1
        if self.since_save >= 10:                 # alle 10 Entscheidungen automatisch in die Datei schreiben
            self.since_save = 0
            self.save_to_file(quiet=True)
        self.show_current()
        self._refresh_stats()

    def skip(self):
        fid = self.queue.current
        self.queue.skip()
        if fid is not None:
            self.notify(f"Standort {self.feats[fid]['site_id']} übersprungen - kommt am Ende wieder")
        self.show_current()

    def back(self):
        res = self.queue.back()
        if res:
            self._write(res[0], res[1], res[2])
            f = self.feats[res[0]]
            self.notify(f"Zurück bei {f['site_id']} (Rang {f['rank']}): Eintrag gelöscht"
                        + (f", war vorher: {res[1]}" if res[1] else ""))
        else:
            self.notify("Nichts mehr zum Zurücknehmen")
        self.show_current()

    def open_map(self):
        fid = self.queue.current
        if fid is not None:
            QDesktopServices.openUrl(QUrl(self.feats[fid]["maps_url"]))

    def draw_part(self):
        if self.drawn is None or self.queue.current is None:
            return
        note = self.note.currentText().strip()
        fid = self.queue.mark("partial", note if note else None)
        self._write(fid, "partial", note if note else None)
        self.notify(f"Standort {self.feats[fid]['site_id']} als PARTIAL eingetragen - jetzt den Umriss zeichnen")
        self.awaiting_draw = True
        self.iface.setActiveLayer(self.drawn)
        self.drawn.startEditing()
        self.iface.actionAddFeature().trigger()
        self.info.setText(self.info.text() + "<br><b>Jetzt den Umriss der Halde zeichnen, kind = dump, dann 'Weiter'.</b>")

    def next_after_draw(self):
        if self.drawn is not None:
            self.drawn.commitChanges(False)
            if not self.drawn.isEditable():
                self.drawn.startEditing()
        self.awaiting_draw = False
        self.iface.setActiveLayer(self.layer)
        self.queue.advance()
        self.show_current()

    # =====================================================================================
    # Gespeicherte KI-Ergebnisse (Spalten ai_*)
    # =====================================================================================
    def _feat(self, fid):
        return self.layer.getFeature(fid)

    def _fld(self, feat, name):
        i = feat.fields().indexOf(name)
        v = feat[name] if i >= 0 else None
        return None if v is None or str(v) in ("NULL", "") else v

    def stored_result(self, fid):
        """(Ergebnis, ai_decision) aus der Datei, oder (None, None)."""
        f = self._feat(fid)
        res = result_from_fields(self._fld(f, "ai_label"), self._fld(f, "ai_dump_p"), self._fld(f, "ai_loc"),
                                 self._fld(f, "ai_reason"))
        if res is None:
            return None, None
        res["context"] = feature_context(f)
        return res, self._fld(f, "ai_decision")

    def store_fields(self, fid, res, decision):
        """KI-Einschätzung samt Entscheidung in die Schicht schreiben (Commit macht der Aufrufer)."""
        fields = self.layer.fields()
        for name, value in (("ai_label", res["label"]), ("ai_dump_p", round(res["dump_probability"], 3)),
                            ("ai_loc", res["dump_location"]), ("ai_reason", res["reason"]),
                            ("ai_decision", decision)):
            self.layer.changeAttributeValue(fid, fields.indexOf(name), value)

    def set_decision(self, fid, decision):
        self.layer.changeAttributeValue(fid, self.layer.fields().indexOf("ai_decision"), decision)
        self._commit()

    def _followup(self, fid, human_review):
        """Du hast per Taste entschieden: bei Standorten mit KI-Ergebnis vermerken, ob ihr übereinstimmt."""
        res, prev = self.stored_result(fid)
        if res is None:
            return
        new = human_followup(prev, human_review, verdict_to_review(res, self.settings["thr_dump"],
                                                                    self.settings["thr_possible"]))
        if new:
            self.set_decision(fid, new)

    def _refresh_stats(self):
        """Überblick über alle KI-Ergebnisse der Schicht (Dump / Möglich / Kein Dump, automatisch
        übersprungen, Stichproben) und die Trefferquote der Stichproben."""
        fields = self.layer.fields()
        i, j = fields.indexOf("ai_decision"), fields.indexOf("ai_dump_p")
        decisions, entries = [], []
        for f in self.layer.getFeatures():
            d = f["ai_decision"] if i >= 0 else None
            d = str(d) if d is not None and str(d) != "NULL" else ""
            if d:
                decisions.append(d)
            pv = f["ai_dump_p"] if j >= 0 else None
            if pv is not None and str(pv) != "NULL":
                entries.append((float(pv), d))
        self.overview.setText(fmt_overview(ai_overview(entries, self.settings["thr_dump"],
                                                        self.settings["thr_possible"])) if entries else "")
        self.stats.setText(audit_summary(decisions))

    # =====================================================================================
    # KI-Teil: Anzeige, Anfragen, Auto-Modus
    # =====================================================================================
    def _cost_text(self, res=None):
        tot = self.total
        line = f"Σ {tot['n']} Checks " + ("(Preis unbekannt)" if tot["unknown"] and not tot["usd"] else fmt_cost(tot["usd"]))
        if res is None or res.get("cached"):
            return line
        usd = cost_of(self.settings["model"], res, self.settings["price_in"], self.settings["price_out"])
        cache = res["usage"]["cache_read"]
        return (f"{res['tokens_in']:,} ein / {res['tokens_out']:,} aus"
                + (f" (davon {cache:,} aus dem Cache)" if cache else "") + f"  {fmt_cost(usd)}\n" + line)

    def add_total(self, res):
        usd = cost_of(self.settings["model"], res, self.settings["price_in"], self.settings["price_out"])
        self.total["n"] += 1
        if usd is None:
            self.total["unknown"] = True
        else:
            self.total["usd"] += usd
        return usd or 0.0

    def _result_for(self, fid):
        if fid in self.results:
            return self.results[fid], None
        res, decision = self.stored_result(fid)
        if res is not None:
            self.results[fid] = res
        return res, decision

    def _refresh_view(self):
        fid = self.queue.current
        if fid is None or fid == self.shown:
            return
        res, _ = self._result_for(fid)
        if res is not None or fid in self.errors:
            self._show(fid)
        else:
            self.pending = None
            for b in (self.accept_btn, self.deny_btn, self.banner):
                b.hide()
            self.cost.setText(self._cost_text())
            self.out.setText("Frage Claude ..." if fid in self.engine.inflight else "")

    def _show(self, fid):
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
        text, color = banner(res, self.settings["thr_dump"], self.settings["thr_possible"])
        self.banner.setText(text)
        self.banner.setStyleSheet(f"background: {color}; color: white; font-size: 20px; "
                                  "font-weight: bold; border-radius: 6px; padding: 6px;")
        self.banner.show()
        if not res["ok"]:
            self.out.setText(f"Antwort nicht lesbar (stop_reason: {res['stop_reason'] or '?'}). "
                             "Bei max_tokens: Limit in den Einstellungen erhöhen oder Thinking aus / "
                             f"Effort niedriger.<br>{res['reason'] or '(leer)'}")
            return
        _, decision = self.stored_result(fid)
        if decision in (None, "pending", "spot_check"):
            self.pending = {"fid": fid, "res": res}
            self.accept_btn.setEnabled(True)
            self.deny_btn.setEnabled(True)
            self.accept_btn.show()
            self.deny_btn.show()
        zone = ""
        if res.get("radius_m"):
            zone = f" &nbsp; <small>Zone: {res['radius_m']:.0f} m" + (f" ({res['context']})" if res.get("context") else "") + "</small>"
        tag = {"auto_clean": "automatisch clean", "spot_check": "Stichprobe - bitte prüfen",
               "accepted": "angenommen", "denied": "abgelehnt"}.get(decision, "")
        self.out.setText(f"<b>{res['label']}</b>" + (f" &nbsp; ({res['confidence']:.2f})" if res["confidence"] else "")
                         + f"<br>Halde: {res['dump_location'] or '-'}{zone}"
                         + (f"<br><b>{tag}</b>" if tag else "") + f"<br>{res['reason']}")

    def _request(self, fid, key):
        feature = self.feats[fid]
        self.engine.submit(self.layer, feature, key, self._on_result)

    def _on_result(self, fid, res, err, meta):
        if err is not None:
            self.errors[fid] = err
        else:
            self.results[fid] = res
            self.add_total(res)
            if res["ok"]:
                self.store_fields(fid, res, "pending")
                self._commit()
        self._refresh_view()

    def classify(self):
        """Knopf 'Einschätzen': aktuellen Standort (neu) einschätzen, auch wenn schon ein Ergebnis da ist."""
        key = self._key()
        fid = self.queue.current
        if not key or fid is None or fid in self.engine.inflight:
            if not key:
                self.out.setText("Kein API-Key.")
            return
        self._ui_changed_silent()
        self.results.pop(fid, None)
        self.errors.pop(fid, None)
        self.shown = None
        self.pending = None
        for b in (self.accept_btn, self.deny_btn, self.banner):
            b.hide()
        self.out.setText("Frage Claude ...")
        try:
            self._request(fid, key)
        except Exception as exc:
            self.errors[fid] = str(exc)
            self._show(fid)

    def _ui_changed_silent(self):
        self.settings["model"] = self.model.currentText().strip() or DEFAULT_MODEL
        self.settings["radius_m"] = float(self.radius.value())
        self._store()

    def _auto_tick(self):
        """Auto-Modus: zeigt das (vorab eingetroffene oder gespeicherte) Ergebnis des aktuellen Standorts und
        stößt Anfragen für den aktuellen sowie die nächsten `prefetch` Standorte an (höchstens eine
        Bildberechnung pro Takt). Nach 'Ablehnen' bleibt derselbe Standort dran, ohne erneute Anfrage."""
        if self.batch is not None or self.awaiting_draw:
            return
        todo = list(self.queue.todo)
        if not todo:
            return
        self._refresh_view()
        if not self.auto.isChecked():
            return
        key = self._key(ask=False)
        if not key:
            return
        n = self.settings["prefetch"]
        for fid in todo[:n + 1]:
            if fid in self.errors or fid in self.engine.inflight:
                continue
            res, _ = self._result_for(fid)
            if res is not None:
                continue
            if len(self.engine.inflight) >= n + 1:
                break
            try:
                self._request(fid, key)
            except Exception as exc:
                self.errors[fid] = str(exc)
            self._refresh_view()
            break

    def decide_ai(self, accepted):
        """Annehmen: KI-Urteil als review eintragen (und weiter); Ablehnen: nur vermerken."""
        pend = self.pending
        if pend is None:
            return
        fid, res = pend["fid"], pend["res"]
        try:
            self.store_fields(fid, res, "accepted" if accepted else "denied")
            self._commit()
        except Exception as exc:
            self.out.setText(f"Speichern fehlgeschlagen: {exc}")
            return
        value = verdict_to_review(res, self.settings["thr_dump"], self.settings["thr_possible"])
        self.pending = None
        self.accept_btn.hide()
        self.deny_btn.hide()
        self.banner.hide()
        if accepted:
            if fid == self.queue.current:
                self.note.setEditText(ai_note(res))
                self.decide(value)
            else:
                self._write(fid, value, ai_note(res))
        self.save_to_file(quiet=True)
        self.out.setText(f"Angenommen: review = {value}" if accepted else "Abgelehnt (review bleibt leer)")

    def to_note(self):
        res = self.results.get(self.queue.current)
        if res:
            self.note.setEditText(ai_note(res))

    # =====================================================================================
    # Stapel
    # =====================================================================================
    def start_batch(self, mode):
        if self.batch is not None:
            return
        key = self._key()
        if not key:
            self.batch_info.setText("Kein API-Key.")
            return
        self._ui_changed_silent()
        self.settings["batch_n"] = self.batch_n.value()
        self._store()
        self.batch = BatchRunner(self, mode, key)
        if not self.batch.fids:
            self.batch_info.setText("Nichts zu tun: keine passenden Standorte." if mode == "batch"
                                    else "Test: keine Standorte ausgewählt (in QGIS Standorte markieren).")
            self.batch = None
            return
        self.batch_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.cancel_btn.show()
        self.progress.setRange(0, len(self.batch.fids))
        self.progress.setValue(0)
        self.progress.show()
        self.batch.start()

    def cancel_batch(self):
        if self.batch is not None:
            self.batch.cancel()

    def batch_finished(self, summary_text):
        self.batch = None
        self.cancel_btn.hide()
        self.progress.hide()
        self.batch_btn.setEnabled(True)
        self.test_btn.setEnabled(True)
        self.batch_info.setText(summary_text)
        self._commit()
        self.save_to_file(quiet=True)
        self.shown = None
        self.show_current()
        self._refresh_stats()

    # =====================================================================================
    def closeEvent(self, event):
        try:
            self.timer.stop()
            if self.batch is not None:
                self.batch.cancel()
            for layer in (self.layer, self.drawn):
                try:
                    if layer is not None and layer.isEditable():
                        layer.commitChanges()
                except RuntimeError:
                    pass
            self.save_to_file(quiet=True)
        except Exception:
            pass
        super().closeEvent(event)
