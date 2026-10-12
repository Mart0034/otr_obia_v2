"""Einstellungsdialog (Modell, Tokens, Prompt, Umkreis, Auto-Clean, Stapel, Preise ...)."""

from .core import DEFAULT_SETTINGS, EFFORTS, CONTEXTS, MODELS, _LIMITS, clean_settings
from .qtcompat import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit,
                       QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QDialogButtonBox, standard_buttons)


def _spin(lo, hi, value, suffix="", step=None):
    w = QSpinBox()
    w.setRange(int(lo), int(hi))
    w.setValue(int(value))
    if suffix:
        w.setSuffix(suffix)
    if step:
        w.setSingleStep(step)
    return w


def _dspin(lo, hi, value, step=0.05, decimals=2):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    return w


class SettingsDialog(QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle("OTR Review - Einstellungen")
        self.resize(600, 760)
        s = settings
        form = QFormLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(MODELS)
        self.model.setEditText(s["model"])
        form.addRow("Modell", self.model)
        self.max_tokens = _spin(*_LIMITS["max_tokens"][:2], s["max_tokens"])
        form.addRow("Max. Ausgabe-Tokens", self.max_tokens)
        self.effort = QComboBox()
        self.effort.addItems(["(Standard)", "low", "medium", "high", "xhigh", "max"])
        self.effort.setCurrentIndex(EFFORTS.index(s["effort"]))
        form.addRow("Effort (Denktiefe)", self.effort)
        self.thinking = QComboBox()
        self.thinking.addItems(["default (Modell entscheidet)", "off (kein / wenig Thinking)"])
        self.thinking.setCurrentIndex(1 if s["thinking"] == "off" else 0)
        form.addRow("Thinking", self.thinking)
        self.use_temp = QCheckBox("Temperatur setzen (sonst API-Standard)")
        self.use_temp.setChecked(s["temperature"] is not None)
        self.temp = _dspin(0.0, 1.0, s["temperature"] if s["temperature"] is not None else 0.0, 0.1)
        self.temp.setEnabled(self.use_temp.isChecked())
        self.use_temp.toggled.connect(self.temp.setEnabled)
        form.addRow(self.use_temp, self.temp)
        self.prompt_cache = QCheckBox("System-Prompt zwischenspeichern (Cache; spart ein paar Prozent, wirkt ab 512 Token)")
        self.prompt_cache.setChecked(s["prompt_cache"])
        form.addRow(self.prompt_cache)

        self.radius = _spin(0, 500, s["radius_m"], " m")
        form.addRow("Umkreis (ohne Umgebungsspalte)", self.radius)
        self.auto_radius = QCheckBox("Umkreis automatisch nach Umgebung (Spalte context)")
        self.auto_radius.setChecked(s["auto_radius"])
        form.addRow(self.auto_radius)
        self.r_inside = _spin(0, 500, s["radius_inside"], " m")
        self.r_near = _spin(0, 500, s["radius_near"], " m")
        self.r_desert = _spin(0, 500, s["radius_desert"], " m")
        form.addRow("  Umkreis im Betrieb (inside_op)", self.r_inside)
        form.addRow("  Umkreis am Rand (near_op)", self.r_near)
        form.addRow("  Umkreis offene Wüste (open_desert)", self.r_desert)
        self.render_mode = QComboBox()
        self.render_mode.addItems(["clean - nur Satellitenbild + Umriss/Umkreis (schnell, im Hintergrund)",
                                   "visible - alle sichtbaren Schichten wie im Kartenfenster"])
        self.render_mode.setCurrentIndex(1 if s["render_mode"] == "visible" else 0)
        form.addRow("Bild für die KI", self.render_mode)
        self.use_examples = QCheckBox("Beispielbilder mitschicken (Menü: 'Beispiele …' im Fenster; kleine Zusatzkosten dank Cache)")
        self.use_examples.setChecked(s["use_examples"])
        form.addRow(self.use_examples)
        self.max_examples = _spin(*_LIMITS["max_examples"][:2], s["max_examples"])
        form.addRow("  höchstens so viele Beispiele", self.max_examples)
        self.example_px = _spin(*_LIMITS["example_px"][:2], s["example_px"], " px", 128)
        form.addRow("  Bildgröße der Beispiele", self.example_px)
        self.tell_context = QCheckBox("Der KI die Umgebung nennen (Betrieb / Wüste) - AUS empfohlen: drückte die Einschätzung "
                                      "in Betrieben Richtung 'clean_mine'")
        self.tell_context.setChecked(s["tell_context"])
        form.addRow(self.tell_context)
        self.highlight_fill = QCheckBox("Standort im Bild gelb füllen (aus = nur Umriss; die Füllung färbt die Reifen ein)")
        self.highlight_fill.setChecked(s["highlight_fill"])
        form.addRow(self.highlight_fill)
        self.ring = QCheckBox("Umkreis gestrichelt einzeichnen (der Prompt erwähnt die Linie)")
        self.ring.setChecked(s["draw_ring"])
        form.addRow(self.ring)
        self.px = _spin(*_LIMITS["max_image_px"][:2], s["max_image_px"], " px", 128)
        form.addRow("Bildgröße (lange Kante)", self.px)
        self.min_view = _spin(*[int(v) for v in _LIMITS["min_view_m"][:2]], s["min_view_m"], " m")
        form.addRow("Kleinster Ausschnitt", self.min_view)
        self.timeout = _spin(*_LIMITS["timeout_s"][:2], s["timeout_s"], " s")
        form.addRow("Zeitlimit API", self.timeout)

        self.thr_dump = _dspin(0.0, 1.0, s["thr_dump"])
        self.thr_possible = _dspin(0.0, 1.0, s["thr_possible"])
        form.addRow("Ab hier 'Dump found'", self.thr_dump)
        form.addRow("Ab hier 'Possible dump'", self.thr_possible)

        self.prefetch = _spin(*_LIMITS["prefetch"][:2], s["prefetch"])
        self.prefetch.setToolTip("Auto-Modus: so viele Standorte vor dem aktuellen werden parallel vorab eingeschätzt.")
        form.addRow("Vorab einschätzen (Standorte)", self.prefetch)

        self.auto_clean = QCheckBox("Stapel: sichere Standorte automatisch als clean eintragen")
        self.auto_clean.setChecked(s["auto_clean"])
        form.addRow(self.auto_clean)
        self.auto_p = _dspin(0.0, 0.5, s["auto_clean_p"], 0.01)
        form.addRow("  ... wenn Halden-Wahrscheinlichkeit höchstens", self.auto_p)
        self.auto_ctx = {c: QCheckBox(c) for c in CONTEXTS}
        for c, box in self.auto_ctx.items():
            box.setChecked(c in s["auto_clean_contexts"].split(","))
            form.addRow(f"  ... in Umgebung" if c == CONTEXTS[0] else "", box)
        self.spot = _spin(*_LIMITS["spot_check_every"][:2], s["spot_check_every"])
        self.spot.setToolTip("Jeder n-te Auto-Clean-Kandidat kommt zur Kontrolle in deine Warteschlange (1 = alle).")
        form.addRow("  Stichprobe: jeder n-te zur Kontrolle", self.spot)
        self.b_parallel = _spin(*_LIMITS["batch_parallel"][:2], s["batch_parallel"])
        form.addRow("Stapel: gleichzeitige Standorte", self.b_parallel)
        self.b_render = _spin(*_LIMITS["batch_render_parallel"][:2], s["batch_render_parallel"])
        form.addRow("Stapel: davon gleichzeitig rendern", self.b_render)
        self.b_budget = _dspin(0.0, 1000.0, s["batch_budget_usd"], 0.25)
        form.addRow("Stapel: Kostenlimit in $ (0 = keins)", self.b_budget)

        self.price_in = _dspin(0.0, 1000.0, s["price_in"], 0.05, 3)
        self.price_out = _dspin(0.0, 1000.0, s["price_out"], 0.05, 3)
        form.addRow("Preis Eingabe $/Mio. Token (0 = Tabelle)", self.price_in)
        form.addRow("Preis Ausgabe $/Mio. Token (0 = Tabelle)", self.price_out)
        self.log_path = QLineEdit(s["log_path"])
        form.addRow("Log-Datei", self.log_path)
        self.prompt = QPlainTextEdit(s["system_prompt"])
        reset = QPushButton("Prompt auf Standard zurücksetzen")
        reset.clicked.connect(lambda _=False: self.prompt.setPlainText(DEFAULT_SETTINGS["system_prompt"]))
        buttons = QDialogButtonBox(standard_buttons("Ok", "Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("System-Prompt (die Antwort muss ein JSON mit label / confidence / "
                                "dump_probability / dump_location / reason bleiben):"))
        layout.addWidget(self.prompt, 1)
        layout.addWidget(reset)
        layout.addWidget(buttons)

    def values(self):
        """Nur die Werte, die dieser Dialog verwaltet (geprüft); andere Einstellungen (auto_next, batch_n)
        bleiben beim Zusammenführen unberührt."""
        raw = {
            "model": self.model.currentText(), "max_tokens": self.max_tokens.value(),
            "effort": EFFORTS[self.effort.currentIndex()],
            "thinking": "off" if self.thinking.currentIndex() == 1 else "default",
            "temperature": self.temp.value() if self.use_temp.isChecked() else None,
            "prompt_cache": self.prompt_cache.isChecked(),
            "system_prompt": self.prompt.toPlainText(),
            "radius_m": self.radius.value(), "auto_radius": self.auto_radius.isChecked(),
            "radius_inside": self.r_inside.value(), "radius_near": self.r_near.value(),
            "radius_desert": self.r_desert.value(), "draw_ring": self.ring.isChecked(),
            "max_image_px": self.px.value(), "min_view_m": self.min_view.value(),
            "timeout_s": self.timeout.value(), "thr_dump": self.thr_dump.value(),
            "thr_possible": self.thr_possible.value(), "prefetch": self.prefetch.value(),
            "auto_clean": self.auto_clean.isChecked(), "auto_clean_p": self.auto_p.value(),
            "auto_clean_contexts": ",".join(c for c, b in self.auto_ctx.items() if b.isChecked()),
            "spot_check_every": self.spot.value(), "batch_parallel": self.b_parallel.value(),
            "batch_render_parallel": self.b_render.value(),
            "render_mode": "visible" if self.render_mode.currentIndex() == 1 else "clean",
            "use_examples": self.use_examples.isChecked(), "max_examples": self.max_examples.value(),
            "example_px": self.example_px.value(), "tell_context": self.tell_context.isChecked(), "highlight_fill": self.highlight_fill.isChecked(),
            "batch_budget_usd": self.b_budget.value(), "price_in": self.price_in.value(),
            "price_out": self.price_out.value(), "log_path": self.log_path.text()}
        cleaned = clean_settings(raw)
        return {k: cleaned[k] for k in raw}
