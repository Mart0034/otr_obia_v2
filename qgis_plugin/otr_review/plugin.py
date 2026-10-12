"""Menü, Werkzeugleiste und Lebenszyklus des Plugins."""

import json

from qgis.core import QgsSettings

from .core import OLD_PREFS_KEY, PREFS_KEY, clean_settings, load_settings
from .layers import layer_by_name, load_candidate_sites
from .qtcompat import (QAction, QColor, QFileDialog, QIcon, QMessageBox, QPainter, QPen, QPixmap, QPointF, enum,
                       exec_dialog, qt)
from .settings_dialog import SettingsDialog

MENU = "&OTR Review"


def make_icon():
    """Kleines Reifen-Symbol, gezeichnet (keine Bilddatei nötig)."""
    pm = QPixmap(32, 32)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(enum(QPainter, "RenderHint", "Antialiasing"))
    p.setPen(QPen(QColor("#222222"), 6))
    p.drawEllipse(QPointF(16, 16), 10, 10)
    p.setPen(QPen(QColor("#e4572e"), 2))
    p.drawEllipse(QPointF(16, 16), 4, 4)
    p.end()
    return QIcon(pm)


class OtrReviewPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.actions = []
        self.panel = None

    # ---- QGIS-Lebenszyklus ------------------------------------------------------------
    def initGui(self):
        icon = make_icon()
        for text, func in (("Kandidatenliste laden …", self.load_list),
                           ("Review-Fenster öffnen", self.open_panel),
                           ("Beispielbilder verwalten …", self.manage_examples),
                           ("Einstellungen …", self.open_settings)):
            action = QAction(icon, text, self.iface.mainWindow())
            action.triggered.connect(lambda _=False, f=func: f())
            self.iface.addPluginToMenu(MENU, action)
            self.actions.append(action)
        self.iface.addToolBarIcon(self.actions[1])         # Knopf: Review-Fenster öffnen

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(MENU, action)
        self.iface.removeToolBarIcon(self.actions[1])
        self.actions = []
        self._drop_panel()

    # ---- Aktionen ---------------------------------------------------------------------
    def _drop_panel(self):
        if self.panel is not None:
            try:
                self.panel.close()
                self.iface.removeDockWidget(self.panel)
                self.panel.deleteLater()
            except RuntimeError:
                pass
            self.panel = None

    def load_list(self):
        path = QFileDialog.getOpenFileName(self.iface.mainWindow(), "candidate_sites.gpkg wählen", "",
                                           "GeoPackage (*.gpkg)")[0]
        if not path:
            return
        self._drop_panel()
        try:
            layer, messages = load_candidate_sites(self.iface, path)
        except Exception as exc:
            QMessageBox.warning(self.iface.mainWindow(), "OTR Review", f"Laden fehlgeschlagen: {exc}")
            return
        for m in messages:
            self.iface.messageBar().pushWarning("OTR Review", m)
        self.open_panel()

    def open_panel(self):
        layer = layer_by_name("candidate_sites")
        if layer is None:
            self.load_list()
            return
        if self.panel is not None and self.panel.isVisible():
            self.panel.raise_()
            return
        self._drop_panel()
        from .panel import OtrPanel
        try:
            self.panel = OtrPanel(self.iface, layer)
        except Exception as exc:
            self.panel = None
            QMessageBox.warning(self.iface.mainWindow(), "OTR Review",
                                f"Das Fenster ließ sich nicht öffnen: {exc}\n\nBitte diese Meldung weitergeben.")
            raise

    def manage_examples(self):
        if self.panel is not None and self.panel.isVisible():
            self.panel.manage_examples()
        else:
            QMessageBox.information(self.iface.mainWindow(), "OTR Review",
                                    "Zuerst das Review-Fenster öffnen (Kandidatenliste laden).")

    def open_settings(self):
        if self.panel is not None and self.panel.isVisible():
            self.panel.open_settings()
            return
        qs = QgsSettings()
        settings = load_settings(qs.value(PREFS_KEY, "") or qs.value(OLD_PREFS_KEY, ""))
        dlg = SettingsDialog(self.iface.mainWindow(), clean_settings(settings))
        if exec_dialog(dlg):
            qs.setValue(PREFS_KEY, json.dumps(clean_settings({**settings, **dlg.values()})))
