"""
Review-Leiste für Kandidaten-Standorte in QGIS
==============================================

Geht die Standorte der Reihe nach durch (Rang 1, 2, 3, ...): zoomt automatisch
auf den nächsten ungeprüften Standort, ein Klick (oder eine Zifferntaste)
trägt das Urteil ein und springt zum nächsten.

Voraussetzung: die Schichten `candidate_sites` und `drawn_polygons` sind geladen
(Skript load_candidate_sites.py zuerst ausführen). Dann dieses Skript in der
QGIS-Python-Konsole öffnen und ausführen: rechts erscheint das Fenster
"Review". Jede Entscheidung wird sofort in die Datei gespeichert.

    1  Dump      echte Reifenhalde, Umriss passt grob
    2  Clean     keine Halde (Fehlalarm, z.B. Stadt, Straße)
    3  Partial   Halde enthalten, aber der Umriss ist viel größer
    4  Unsure    unklar
    5  Skip      später nochmal ansehen (kommt am Ende wieder)
    S  Sichern   (Knopf) schreibt alles in die .gpkg-Datei; passiert auch automatisch alle
                 10 Entscheidungen und beim Schließen des Fensters. Vor dem Hochladen/
                 Kopieren der Datei einmal drücken.
    6  Back      letzte Entscheidung zurücknehmen (beliebig oft: geht Schritt für
                 Schritt zurück, solange das Fenster offen ist)
    7  Karte     in Google Maps öffnen
   Notiz     im Feld unter "Zurück" eine Notiz wählen oder tippen (town, yard,
             road, ...), dann Entscheidung drücken - sie wird mitgespeichert.
             Nach Auswahl aus der Liste bzw. Enter gelten die Zifferntasten wieder.
   "Teil einzeichnen": setzt Partial und schaltet die Zeichen-Schicht ein - Umriss
   der echten Halde zeichnen (Rechtsklick beendet), kind = dump wählen, dann
   "Weiter".

Die Zifferntasten 1-6 gelten im ganzen QGIS-Fenster; beim Eintippen von Zahlen in
Tabellenzellen deshalb zuerst das Review-Fenster anklicken.

Hinweis: gegen die PyQGIS-3-Schnittstelle geschrieben, nicht in QGIS getestet;
die Reihenfolge-Logik (ReviewQueue) ist getestet. Meldet ein Schritt einen
Fehler, bitte die Meldung weitergeben.
"""

from collections import deque


class ReviewQueue:
    """Reihenfolge-Logik ohne QGIS: ungeprüfte Standorte nach Rang, Überspringen
    hängt hinten an, Zurück stellt die letzte Entscheidung wieder vorne ein."""

    def __init__(self, items):
        """items: Liste von (fid, rank, review) oder (fid, rank, review, note);
        leeres review = noch ungeprüft."""
        ordered = sorted(items, key=lambda it: it[1])
        self.total = len(ordered)
        self.reviews = {it[0]: (it[2] or "") for it in ordered}
        self.notes = {it[0]: ((it[3] if len(it) > 3 else "") or "") for it in ordered}
        self.todo = deque(it[0] for it in ordered if not it[2])
        self.history = []                      # (fid, vorheriges Urteil, vorherige Notiz)

    @property
    def current(self):
        return self.todo[0] if self.todo else None

    @property
    def done(self):
        return self.total - len(self.todo)

    def mark(self, value, note=None):
        """Urteil (und optional eine Notiz) für den aktuellen Standort setzen, ohne
        weiterzugehen. note=None lässt die bisherige Notiz unverändert."""
        fid = self.current
        if fid is None:
            return None
        self.history.append((fid, self.reviews[fid], self.notes[fid]))
        self.reviews[fid] = value
        if note is not None:
            self.notes[fid] = note
        return fid

    def advance(self):
        """Zum nächsten Standort; ein bereits beurteilter fällt aus der Liste."""
        if not self.todo:
            return None
        fid = self.todo.popleft()
        if not self.reviews[fid]:              # nichts eingetragen: später wieder (Skip)
            self.todo.append(fid)
        return self.current

    def decide(self, value, note=None):
        fid = self.mark(value, note)
        self.advance()
        return fid

    def skip(self):
        return self.advance()

    def back(self):
        """Letzte Entscheidung zurücknehmen. Gibt (fid, altes Urteil, alte Notiz) zurück."""
        if not self.history:
            return None
        fid, old, old_note = self.history.pop()
        self.reviews[fid] = old
        self.notes[fid] = old_note
        if fid in self.todo:
            self.todo.remove(fid)
        self.todo.appendleft(fid)
        return fid, old, old_note

    def counts(self):
        out = {}
        for v in self.reviews.values():
            if v:
                out[v] = out.get(v, 0) + 1
        return out


try:  # außerhalb von QGIS (Tests) nicht verfügbar
    from qgis.core import QgsCoordinateTransform, QgsProject, QgsRectangle
    from qgis.PyQt.QtCore import Qt, QUrl
    from qgis.PyQt.QtGui import QDesktopServices, QKeySequence
    from qgis.PyQt.QtWidgets import QComboBox, QDockWidget, QGridLayout, QLabel, QPushButton, QWidget
    try:                                  # Qt6: QShortcut liegt in QtGui, Qt5: in QtWidgets
        from qgis.PyQt.QtGui import QShortcut
    except ImportError:
        from qgis.PyQt.QtWidgets import QShortcut
    IN_QGIS = True
except ImportError:
    IN_QGIS = False
    QDockWidget = object   # damit die Klasse unten auch ohne QGIS (Tests) importierbar bleibt

MIN_VIEW_M = 400.0   # kleinster Kartenausschnitt (Meter) beim Heranzoomen
NOTE_PRESETS = ["", "town", "yard / equipment", "road / haul road", "pit / rock pile", "neat dump", "messy dump", "rubber, unclear"]


def _qt_enum(group_name, name):
    """Qt6 (PyQt6) kennt nur Qt.<Gruppe>.<Name>, Qt5 nur Qt.<Name> - beides unterstützen."""
    group = getattr(Qt, group_name, None)
    if group is not None and hasattr(group, name):
        return getattr(group, name)
    return getattr(Qt, name)


def _layer(name):
    project = QgsProject.instance()
    found = project.mapLayersByName(name)
    if not found:  # auch Schichten mit leicht anderem Namen (z.B. umbenannt) über die Quelle finden
        found = [l for l in project.mapLayers().values()
                 if l.name().lower().startswith(name) or f"layername={name}" in l.source()]
    if not found:
        have = [l.name() for l in project.mapLayers().values()]
        raise RuntimeError(f"Schicht '{name}' nicht gefunden - zuerst load_candidate_sites.py ausführen. "
                           f"Vorhandene Schichten: {have}")
    return found[0]


class ReviewPanel(QDockWidget):  # pragma: no cover - braucht QGIS
    DECISIONS = (("1  Dump", "dump", "1"), ("2  Clean", "clean", "2"),
                 ("3  Partial", "partial", "3"), ("4  Unsure", "unsure", "4"))

    def __init__(self, iface):
        super().__init__("Review", iface.mainWindow())
        self.iface = iface
        self.setFocusPolicy(_qt_enum("FocusPolicy", "StrongFocus"))     # damit setFocus() greift
        self.layer = _layer("candidate_sites")
        self.drawn = None
        try:
            self.drawn = _layer("drawn_polygons")
        except RuntimeError:
            pass
        self.idx = self.layer.fields().indexOf("review")
        self.note_idx = self.layer.fields().indexOf("note")
        feats = {f.id(): f for f in self.layer.getFeatures()}
        self.feats = feats
        self.queue = ReviewQueue([(fid, f["rank"], f["review"] if f["review"] else "",
                                   f["note"] if f["note"] else "") for fid, f in feats.items()])
        self.awaiting_draw = False

        body = QWidget()
        grid = QGridLayout(body)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        grid.addWidget(self.info, 0, 0, 1, 2)
        self.last = QLabel("")                    # was zuletzt entschieden wurde
        self.last.setWordWrap(True)
        self.last.setStyleSheet("font-weight: bold; color: #1a7f37;")
        grid.addWidget(self.last, 1, 0, 1, 2)
        back_btn = QPushButton("6  ◀ Zurück  (letzte Entscheidung rückgängig)")
        back_btn.setMinimumHeight(34)
        back_btn.clicked.connect(lambda _=False: self.back())
        grid.addWidget(back_btn, 2, 0, 1, 2)
        # Notiz: aus der Liste wählen oder frei tippen; wird mit der nächsten Entscheidung gespeichert
        self.note = QComboBox()
        self.note.setEditable(True)
        self.note.addItems(NOTE_PRESETS)
        self.note.setCurrentIndex(0)
        self.note.lineEdit().setPlaceholderText("Notiz (z.B. town, yard, road) - dann Entscheidung wählen")
        self.note.activated.connect(lambda _=0: self.setFocus())          # danach gelten die Zifferntasten wieder
        self.note.lineEdit().returnPressed.connect(self.setFocus)
        grid.addWidget(self.note, 3, 0, 1, 2)
        for n, (text, value, _) in enumerate(self.DECISIONS):
            btn = QPushButton(text)
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda _=False, v=value: self.decide(v))
            grid.addWidget(btn, 4 + n // 2, n % 2)
        extra = [("5  Skip", self.skip), ("7  Karte", self.open_map),
                 ("Teil einzeichnen", self.draw_part), ("Weiter", self.next_after_draw),
                 ("💾 Sichern", self.save_to_file)]
        for n, (text, func) in enumerate(extra):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, f=func: f())
            grid.addWidget(btn, 6 + n // 2, n % 2)
        self.setWidget(body)
        for key, func in (("1", lambda: self.decide("dump")), ("2", lambda: self.decide("clean")),
                          ("3", lambda: self.decide("partial")), ("4", lambda: self.decide("unsure")),
                          ("5", self.skip), ("6", self.back), ("7", self.open_map)):
            QShortcut(QKeySequence(key), self, activated=func, context=_qt_enum("ShortcutContext", "ApplicationShortcut"))
        iface.addDockWidget(_qt_enum("DockWidgetArea", "RightDockWidgetArea"), self)
        self.layer.startEditing()
        self.show_current()

    # ---- Darstellung -------------------------------------------------
    def show_current(self):
        fid = self.queue.current
        counts = ", ".join(f"{k} {v}" for k, v in sorted(self.queue.counts().items())) or "noch nichts"
        if fid is None:
            self.layer.removeSelection()
            self.info.setText(f"Fertig. Alle {self.queue.total} Standorte haben ein Urteil.\n{counts}")
            return
        f = self.feats[fid]
        self.info.setText(
            f"<b>{f['site_id']}</b> &nbsp; Rang {f['rank']}<br>{f['mine_id']} &nbsp; {f['size_class']}<br>"
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
        """Kurze Rückmeldung: im Fenster und als Meldung oben in der Karte."""
        self.last.setText(text)
        try:
            bar = self.iface.messageBar()
            bar.clearWidgets()
            bar.pushSuccess("Review", text)
        except Exception:
            pass

    # ---- Aktionen ----------------------------------------------------
    def _write(self, fid, value, note=None):
        self.layer.changeAttributeValue(fid, self.idx, value if value else None)
        if note is not None and self.note_idx >= 0:
            self.layer.changeAttributeValue(fid, self.note_idx, note if note else None)
        self.layer.commitChanges(False)        # sofort speichern, Bearbeitungsmodus bleibt an
        if not self.layer.isEditable():
            self.layer.startEditing()

    def save_to_file(self, quiet=False):
        """Schreibt die in der Seitendatei (-wal) wartenden Änderungen in die eigentliche
        .gpkg-Datei - danach lässt sie sich direkt kopieren/hochladen."""
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
            f = self.feats[fid]
            self.notify(f"Standort {f['site_id']} (Rang {f['rank']}) als {value.upper()} eingetragen"
                        + (f" - Notiz: {note}" if note else ""))
        self.note.setCurrentIndex(0)
        self.note.clearEditText()
        self.since_save = getattr(self, "since_save", 0) + 1
        if self.since_save >= 10:              # alle 10 Entscheidungen automatisch in die Datei schreiben
            self.since_save = 0
            self.save_to_file(quiet=True)
        self.show_current()

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
        self.iface.actionAddFeature().trigger()   # Zeichenwerkzeug "Polygon hinzufügen"
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

    def closeEvent(self, event):
        # Die Schichten können inzwischen entfernt worden sein (z.B. load_candidate_sites.py
        # erneut ausgeführt) - dann gibt es nichts mehr zu speichern.
        for layer in (self.layer, self.drawn):
            try:
                if layer is not None and layer.isEditable():
                    layer.commitChanges()
            except RuntimeError:
                pass
        try:
            self.save_to_file(quiet=True)
        except Exception:
            pass
        super().closeEvent(event)


if IN_QGIS:  # beim Ausführen im QGIS-Editor/Konsole
    try:
        _panel.close()  # noqa: F821  (altes Fenster einer früheren Ausführung)
        _panel.deleteLater()  # noqa: F821
    except Exception:
        pass
    _panel = ReviewPanel(iface)  # noqa: F821  (iface gibt es in der QGIS-Konsole)
