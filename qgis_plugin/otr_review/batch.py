"""Stapel-Analyse: viele Standorte nacheinander von der KI einschätzen lassen.

Modus "batch": die nächsten N noch nicht eingeschätzten, ungeprüften Standorte. Sichere Wüsten-Standorte
(siehe core.auto_clean_ok) werden automatisch als clean eingetragen, jeder n-te davon bleibt als Stichprobe
für dich offen; alles andere bleibt in deiner Warteschlange, mit gespeicherter Einschätzung.
Modus "test": die in QGIS markierten Standorte (z.B. alle mit review = dump / partial) werden
eingeschätzt, ohne etwas zu ändern - zeigt, ob die KI echte Halden je unter die Auto-Clean-Grenze stellt.
"""

from collections import Counter

from .core import ai_note, ai_overview, auto_clean_ok, batch_outcome, fmt_cost, fmt_overview
from .qtcompat import QMessageBox, QTimer


class BatchRunner:
    def __init__(self, panel, mode, key):
        self.panel, self.mode, self.key = panel, mode, key
        self.cfg = dict(panel.settings)
        self.fids = self._pick()
        self.todo = list(self.fids)
        self.done = 0
        self.cost = 0.0
        self.counts = Counter()
        self.consecutive_errors = 0
        self.last_error = ""
        self.tally = []                     # (Wahrscheinlichkeit, Ergebnis) dieses Stapels
        self.rows = []                      # (site_id, bisheriges Urteil, p, label, Ort) im Testmodus
        self.stop_reason = ""
        self.since_commit = 0
        self.timer = QTimer(panel)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self._tick)

    def _pick(self):
        p = self.panel
        if self.mode == "test":
            sel = set(p.layer.selectedFeatureIds())
            return [f for f in sorted(sel, key=lambda x: p.feats[x]["rank"] if x in p.feats else 0) if f in p.feats]
        out = []
        for fid in list(p.queue.todo):
            if fid == p.queue.current:      # der Standort, den du gerade ansiehst, bleibt unberührt
                continue
            res, decision = p.stored_result(fid)
            if res is None and not decision:
                out.append(fid)
            if len(out) >= self.cfg["batch_n"]:
                break
        return out

    # ---------------------------------------------------------------------------------
    def start(self):
        self._progress()
        self.timer.start()

    def cancel(self):
        self.stop_reason = self.stop_reason or "abgebrochen"
        self.todo.clear()

    def _tick(self):
        p = self.panel
        budget = self.cfg["batch_budget_usd"]
        if not self.stop_reason and budget > 0 and self.cost >= budget:
            self.stop_reason = f"Kostenlimit {fmt_cost(budget)} erreicht"
            self.todo.clear()
        if not self.stop_reason and self.consecutive_errors >= 5:
            self.stop_reason = f"5 Fehler in Folge ({self.last_error})"
            self.todo.clear()
        if self.todo and len(p.engine.inflight) < self.cfg["batch_parallel"]:
            fid = self.todo.pop(0)
            try:
                p.engine.submit(p.layer, p.feats[fid], self.key, self._on_result)
            except Exception as exc:
                self._on_result(fid, None, str(exc), {})
        if not self.todo and not p.engine.inflight:
            self._finish()

    def _on_result(self, fid, res, err, meta):
        p = self.panel
        self.done += 1
        if err is not None:
            self.counts["Fehler"] += 1
            self.consecutive_errors += 1
            self.last_error = err[:120]
            if "401" in err or "x-api-key" in err:
                self.stop_reason = "API-Schlüssel abgelehnt"
                self.todo.clear()
        else:
            self.consecutive_errors = 0
            self.cost += p.add_total(res)
            if not res["ok"]:
                self.counts["unlesbar"] += 1
            elif self.mode == "test":
                self.tally.append((res["dump_probability"], "test"))
                self._record_test(fid, res, meta)
            else:
                self._apply(fid, res, meta)
        self._progress()

    def _record_test(self, fid, res, meta):
        f = self.panel.feats[fid]
        self.rows.append((f["site_id"], f["review"] or "-", res["dump_probability"], res["label"],
                          res["dump_location"] or "-"))
        self.counts["eingeschätzt"] += 1

    def _apply(self, fid, res, meta):
        p, f = self.panel, self.panel.feats[fid]
        outcome = batch_outcome(res, meta["context"], f["site_id"], self.cfg)
        self.tally.append((res["dump_probability"], outcome))
        p.store_fields(fid, res, outcome)
        if outcome == "auto_clean":
            note = ai_note(res) + " [auto]"
            p.layer.changeAttributeValue(fid, p.idx, "clean")
            if p.note_idx >= 0:
                p.layer.changeAttributeValue(fid, p.note_idx, note)
            p.queue.set_external(fid, "clean", note)
        self.counts[{"auto_clean": "automatisch clean", "spot_check": "Stichprobe (du prüfst)",
                     "pending": "zu prüfen"}[outcome]] += 1
        self.since_commit += 1
        if self.since_commit >= 25:
            self.since_commit = 0
            p._commit()

    def _overview(self):
        return ai_overview(self.tally, self.cfg["thr_dump"], self.cfg["thr_possible"])

    def _progress(self):
        """Laufende Zählung: so sieht man schon während des Stapels, was gefunden und übersprungen wird."""
        p = self.panel
        p.progress.setValue(self.done)
        o = self._overview()
        extra = "".join(f" · {k} {v}" for k, v in self.counts.items() if k in ("Fehler", "unlesbar"))
        p.batch_info.setText(
            f"{self.done}/{len(self.fids)} · Dump {o['dump']} · Möglich {o['possible']} · Kein Dump {o['none']}"
            + (f" · automatisch übersprungen {o['auto_clean']} · Stichprobe {o['spot_check']}" if self.mode == "batch" else "")
            + extra + f"  ·  Kosten {fmt_cost(self.cost)}")

    def _finish(self):
        self.timer.stop()
        p = self.panel
        o = self._overview()
        head = ("Test" if self.mode == "test" else "Stapel") + f" fertig: {self.done}/{len(self.fids)}"
        if self.stop_reason:
            head += f" ({self.stop_reason})"
        errors = self.counts.get("Fehler", 0) + self.counts.get("unlesbar", 0)
        text = (head + "<br>" + fmt_overview(o) + (f"<br>{errors} ohne Ergebnis (Fehler/unlesbar)" if errors else "")
                + f"<br>Kosten {fmt_cost(self.cost)}")
        if self.mode == "test" and self.rows:
            self._show_test_report()
        p.batch_finished(text)
        if self.mode == "batch" and o["n"]:
            self._show_batch_report(o, head)

    def _show_batch_report(self, o, head):
        """Sofort nach dem Stapel: was die KI gefunden hat und was für dich übrig bleibt."""
        lines = [head, "",
                 f"{o['n']} Standorte eingeschätzt.", "",
                 f"Dump vermutet:           {o['dump']}",
                 f"Möglich / unsicher:      {o['possible']}",
                 f"Kein Dump:               {o['none']}",
                 f"   davon automatisch als clean übersprungen: {o['auto_clean']}",
                 f"   davon als Stichprobe für dich:            {o['spot_check']}", "",
                 f"Für dich zu prüfen (Warteschlange): {o['to_review']}",
                 f"Kosten: {fmt_cost(self.cost)}"]
        box = QMessageBox(self.panel)
        box.setWindowTitle("Stapel-Ergebnis")
        box.setText("\n".join(lines))
        box.exec() if hasattr(box, "exec") else box.exec_()

    def _show_test_report(self):
        cfg = self.cfg
        rows = sorted(self.rows, key=lambda r: r[2])
        low = [r for r in rows if r[2] <= cfg["auto_clean_p"]]
        would = [r for r in rows if auto_clean_ok(
            {"ok": True, "dump_probability": r[2], "dump_location": "" if r[4] == "-" else r[4], "label": r[3]},
            "open_desert", cfg)]
        lines = [f"{len(rows)} Standorte eingeschätzt.",
                 f"Kleinste Halden-Wahrscheinlichkeit: {rows[0][2]:.0%}",
                 f"Höchstens {cfg['auto_clean_p']:.0%}: {len(low)}  (davon würden als Auto-Clean gelten: {len(would)})", "",
                 "Niedrigste Werte (Standort, bisheriges Urteil, p, Label, Ort):"]
        lines += [f"{r[0]}  {r[1]:8}  {r[2]:4.0%}  {r[3]}  {r[4]}" for r in rows[:25]]
        box = QMessageBox(self.panel)
        box.setWindowTitle("Test auf Auswahl")
        box.setText("\n".join(lines))
        box.exec() if hasattr(box, "exec") else box.exec_()
