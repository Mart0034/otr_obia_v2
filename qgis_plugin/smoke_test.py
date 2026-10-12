"""Rauchtest des Plugins in einem echten (headless) QGIS - kein pytest, sondern:

    QT_QPA_PLATFORM=offscreen /usr/bin/python3.12 qgis_plugin/smoke_test.py

Baut eine kleine Standortdatei, öffnet das Fenster gegen eine Attrappe der QGIS-Oberfläche, ersetzt die
Anthropic-API durch eine Attrappe und prüft Review, KI-Einschätzung, Auto-Modus, Stapel und Test-Modus.
Braucht python3-qgis (apt) - in der normalen Testsuite läuft das nicht mit.
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["ANTHROPIC_API_KEY"] = "sk-test"

from osgeo import ogr, osr                                              # noqa: E402
from qgis.core import QgsApplication, QgsProject                         # noqa: E402
from qgis.gui import QgsMapCanvas, QgsMessageBar                         # noqa: E402
from qgis.PyQt.QtCore import QCoreApplication                            # noqa: E402
from qgis.PyQt.QtWidgets import QAction, QMainWindow, QMessageBox        # noqa: E402

app = QgsApplication([], False)
app.initQgis()


class StubIface:
    def __init__(self):
        self.win = QMainWindow()
        self.canvas = QgsMapCanvas(self.win)
        self.canvas.resize(600, 400)
        self.bar = QgsMessageBar(self.win)
        self.act = QAction("add", self.win)
        self.menu, self.toolbar = [], []

    def mainWindow(self): return self.win
    def mapCanvas(self): return self.canvas
    def messageBar(self): return self.bar
    def addDockWidget(self, area, dock): self.win.addDockWidget(area, dock)
    def removeDockWidget(self, dock): self.win.removeDockWidget(dock)
    def addPluginToMenu(self, name, action): self.menu.append(action)
    def removePluginMenu(self, name, action): self.menu.remove(action)
    def addToolBarIcon(self, action): self.toolbar.append(action)
    def removeToolBarIcon(self, action): self.toolbar.remove(action)
    def setActiveLayer(self, layer): pass
    def actionAddFeature(self): return self.act


def make_gpkg(path):
    drv = ogr.GetDriverByName("GPKG")
    ds = drv.CreateDataSource(path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32719)
    lyr = ds.CreateLayer("candidate_sites", srs, ogr.wkbPolygon)
    fields = [("rank", ogr.OFTInteger), ("site_id", ogr.OFTString), ("mine_id", ogr.OFTString),
              ("size_class", ogr.OFTString), ("area_m2", ogr.OFTReal), ("mean_proba", ogr.OFTReal),
              ("maps_url", ogr.OFTString), ("review", ogr.OFTString), ("note", ogr.OFTString),
              ("context", ogr.OFTString)]
    for name, typ in fields:
        lyr.CreateField(ogr.FieldDefn(name, typ))
    contexts = ["open_desert", "open_desert", "inside_op", "open_desert", "open_desert", "near_op", "open_desert", "open_desert"]
    x0, y0 = 400000.0, 7400000.0
    for i, ctx in enumerate(contexts):
        f = ogr.Feature(lyr.GetLayerDefn())
        f["rank"], f["site_id"], f["mine_id"] = i + 1, f"S{i + 1:04d}", "q_way_1"
        f["size_class"], f["area_m2"], f["mean_proba"] = "medium", 3000.0, 0.8
        f["maps_url"], f["context"] = "https://example.com", ctx
        x = x0 + i * 400
        f.SetGeometry(ogr.CreateGeometryFromWkt(
            f"POLYGON(({x} {y0},{x + 60} {y0},{x + 60} {y0 + 60},{x} {y0 + 60},{x} {y0}))"))
        lyr.CreateFeature(f)
    ds = None


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        QCoreApplication.processEvents()
        time.sleep(0.01)


def wait_until(cond, timeout=20, what=""):
    end = time.time() + timeout
    while time.time() < end:
        pump(0.05)
        if cond():
            return
    raise AssertionError(f"Zeitüberschreitung: {what}")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


# ---- API-Attrappe: Wahrscheinlichkeit nach Standort-Nummer im Zonen-Text -------------------------------
import otr_review.core as core                                           # noqa: E402

CALLS = []


def fake_call_api(payload, api_key, timeout=90, opener=None):
    text = payload["messages"][0]["content"][1]["text"]
    CALLS.append(payload)
    assert payload["messages"][0]["content"][0]["source"]["data"], "kein Bild"
    p = 0.04 if "open desert" in text else 0.45
    body = {"label": "clean_desert" if p < 0.1 else "clean_yard", "confidence": 0.9, "dump_probability": p,
            "dump_location": "none" if p < 0.1 else "nearby", "reason": "The zone shows bare sand."}
    return {"content": [{"type": "text", "text": json.dumps(body)}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1500, "output_tokens": 90, "cache_read_input_tokens": 400}}


core.call_api = fake_call_api
QMessageBox.exec = lambda self: print("  (Bericht)\n" + self.text())
if hasattr(QMessageBox, "exec_"):
    QMessageBox.exec_ = QMessageBox.exec

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "sites.gpkg")
make_gpkg(path)
iface = StubIface()

print("Plugin laden:")
from otr_review.layers import load_candidate_sites                      # noqa: E402
from otr_review.panel import OtrPanel                                    # noqa: E402
from otr_review.plugin import OtrReviewPlugin                            # noqa: E402

plugin = OtrReviewPlugin(iface)
plugin.initGui()
check(len(iface.menu) == 3 and len(iface.toolbar) == 1, "Menü (3 Einträge) und Werkzeugleiste")
layer, msgs = load_candidate_sites(iface, path)
check(layer.isValid() and layer.featureCount() == 8, f"Schicht geladen (8 Standorte), Meldungen: {msgs}")
check(QgsProject.instance().mapLayersByName("drawn_polygons"), "Zeichenschicht geladen")

print("Hintergrund-Rendern (clean):")
from osgeo import gdal                                                   # noqa: E402
from qgis.core import QgsRasterLayer                                     # noqa: E402
import numpy as np                                                       # noqa: E402
rpath = os.path.join(tmp, "basemap.tif")
ds = gdal.GetDriverByName("GTiff").Create(rpath, 400, 100, 3, gdal.GDT_Byte)
ds.SetGeoTransform((399900.0, 10.0, 0, 7400200.0, 0, -10.0))
srs = osr.SpatialReference()
srs.ImportFromEPSG(32719)
ds.SetProjection(srs.ExportToWkt())
rng = np.random.default_rng(1)
for b in range(3):
    ds.GetRasterBand(b + 1).WriteArray((120 + rng.integers(0, 40, (100, 400))).astype("uint8"))
ds = None
raster = QgsRasterLayer(rpath, "basemap")
check(raster.isValid(), "Satelliten-Attrappe (GeoTIFF) geladen")
QgsProject.instance().addMapLayer(raster)
iface.canvas.setLayers([layer, raster])                  # sichtbar: Standortschicht UND Satellitenbild
from otr_review.render import raster_layers, start_render_clean            # noqa: E402
check([l.name() for l in raster_layers(iface.canvas)] == ["basemap"], "nur Rasterschichten werden für 'clean' genommen")
feats = [f for f in layer.getFeatures()][:3]
got = {}
for f in feats:                                          # drei Aufträge gleichzeitig starten
    start_render_clean(iface.canvas, layer, f, core.clean_settings({}), 100.0, lambda img, fid=f.id(): got.__setitem__(fid, img))
check(len(got) == 0, "start_render_clean kehrt sofort zurück (nichts blockiert)")
wait_until(lambda: len(got) == 3, what="3 Hintergrund-Bilder")
img = got[feats[0].id()]
c = img.pixelColor(img.width() // 2, img.height() // 2)
check(c.red() > 150 and c.green() > 140 and c.blue() < 120, f"Standort gelb markiert in der Bildmitte (RGB {c.red()},{c.green()},{c.blue()})")
corner = img.pixelColor(5, 5)
check(all(110 <= v <= 170 for v in (corner.red(), corner.green(), corner.blue())) or corner.red() == 255, f"Rand zeigt nur das Satellitenbild, keine Modell-Schichten (RGB {corner.red()},{corner.green()},{corner.blue()})")
open(os.path.join(tmp, "clean.png"), "wb").write(__import__("otr_review.render", fromlist=["image_png"]).image_png(img))
print("   BILD:", os.path.join(tmp, "clean.png"))

print("Fenster:")
panel = OtrPanel(iface, layer)
pump(0.3)
check(layer.fields().indexOf("ai_decision") >= 0 and layer.fields().indexOf("ai_reason") >= 0, "ai_*-Spalten angelegt")
check(panel.queue.current is not None and panel.queue.total == 8, "Warteschlange gebaut")
first = panel.queue.current
panel.decide("clean")
check(layer.getFeature(first)["review"] == "clean", "Taste 'clean' schreibt in die Schicht")
check(panel.queue.current != first, "springt zum nächsten Standort")
panel.back()
check(layer.getFeature(first)["review"] in (None, "") or str(layer.getFeature(first)["review"]) == "NULL", "Zurück nimmt es zurück")

print("KI einzeln:")
panel.classify()
cur = panel.queue.current
wait_until(lambda: cur in panel.results, what="Ergebnis")
res = panel.results[cur]
check(res["ok"] and abs(res["dump_probability"] - 0.04) < 1e-9, "Ergebnis gelesen (open_desert -> 0.04)")
check(res["reason"].startswith("Bare sand"), "Füllwort 'The zone shows' entfernt")
check(panel.banner.text().startswith("No dump"), f"Kasten: {panel.banner.text()}")
check(layer.getFeature(cur)["ai_decision"] == "pending", "Ergebnis in der Datei gespeichert (pending)")
check(CALLS[-1].get("system")[0].get("cache_control"), "Prompt-Cache aktiv")
check(any("Zone radius 100 m" in c["messages"][0]["content"][1]["text"] for c in CALLS), "Wüsten-Umkreis 100 m")
panel.decide_ai(True)
pump(0.2)
check(str(layer.getFeature(cur)["review"]) == "clean" and layer.getFeature(cur)["ai_decision"] == "accepted",
      "Annehmen: review=clean, ai_decision=accepted")

print("Stapel:")
CALLS.clear()
panel.batch_n.setValue(10)
panel.start_batch("batch")
check(panel.batch is not None, f"Stapel gestartet ({len(panel.batch.fids)} Standorte)")
wait_until(lambda: panel.batch is None, timeout=40, what="Stapel fertig")
rows = {f["site_id"]: (str(f["review"]), str(f["ai_decision"])) for f in layer.getFeatures()}
print("   ", rows)
auto = [k for k, v in rows.items() if v[1] == "auto_clean"]
spot = [k for k, v in rows.items() if v[1] == "spot_check"]
check(len(auto) + len(spot) >= 1, f"Auto-Clean {auto} / Stichprobe {spot}")
check(all(rows[k][0] == "clean" for k in auto), "Auto-Clean trägt review=clean ein")
check(all(rows[k][0] in ("NULL", "None", "") for k in spot), "Stichprobe bleibt ungeprüft")
yard = [k for k, v in rows.items() if k in ("S0003", "S0006")]
check(all(rows[k][1] == "pending" for k in yard), "inside_op/near_op werden nie automatisch clean")
check("Auto-Clean" in panel.stats.text(), f"Statistik: {panel.stats.text()}")
check("automatisch übersprungen" in panel.overview.text() and "Kein Dump" in panel.overview.text(),
      f"Überblick: {panel.overview.text()}")
check("automatisch übersprungen" in panel.batch_info.text() or "übersprungen" in panel.batch_info.text(),
      f"Stapel-Zusammenfassung: {panel.batch_info.text()[:140]!r}")
n_calls = len(CALLS)
panel.start_batch("batch")
check(panel.batch is None or not panel.batch.fids, "zweiter Stapel überspringt bereits eingeschätzte Standorte")
check(len(CALLS) == n_calls, "keine zusätzlichen Anfragen")

print("Test-Modus:")
layer.selectByIds([f.id() for f in layer.getFeatures()][:3])
before = {f["site_id"]: str(f["review"]) for f in layer.getFeatures()}
panel.start_batch("test")
wait_until(lambda: panel.batch is None, timeout=40, what="Test fertig")
after = {f["site_id"]: str(f["review"]) for f in layer.getFeatures()}
check(before == after, "Test ändert nichts")

print("Menschliche Entscheidung auf Stichprobe:")
if spot:
    fid = [f.id() for f in layer.getFeatures() if f["site_id"] == spot[0]][0]
    while panel.queue.current != fid and panel.queue.current is not None:
        panel.skip()
    panel.decide("dump")
    check(layer.getFeature(fid)["ai_decision"] == "spot_miss", "Stichprobe als dump bestätigt -> spot_miss")

print("Auto-Modus mit Vorab-Einschätzung:")
CALLS.clear()
panel.results.clear()
panel.errors.clear()
panel.shown = None
for f in layer.getFeatures():                       # frische Standorte ohne gespeicherte Einschätzung
    pass
panel.store_settings(core.clean_settings({**panel.settings, "prefetch": 2}))
for fid in list(panel.queue.todo):                  # gespeicherte Ergebnisse der Test-Standorte löschen
    i = layer.fields().indexOf("ai_decision")
    layer.changeAttributeValue(fid, i, None)
    layer.changeAttributeValue(fid, layer.fields().indexOf("ai_label"), None)
    layer.changeAttributeValue(fid, layer.fields().indexOf("ai_dump_p"), None)
layer.commitChanges(False)
panel.auto.setChecked(True)
todo = list(panel.queue.todo)[:3]
wait_until(lambda: all(f in panel.results for f in todo), timeout=30, what="Vorab-Ergebnisse")
check(len(CALLS) == 3, f"aktueller + 2 Vorab-Standorte eingeschätzt ({len(CALLS)} Anfragen)")
wait_until(lambda: panel.shown == panel.queue.current, what="Anzeige")
check(panel.banner.isVisible() or panel.banner.text() != "", f"Ergebnis des aktuellen Standorts angezeigt: {panel.banner.text()}")
n = len(CALLS)
pump(1.0)
check(len(CALLS) == n, "keine Wiederholung, solange nichts weiterging")
panel.auto.setChecked(False)

print("Gespeicherte Ergebnisse (kein erneuter Aufruf):")
CALLS.clear()
cur = panel.queue.current
panel.results.clear()
panel.shown = None
panel._refresh_view()
check(cur in panel.results and not CALLS, "Ergebnis aus der Datei geladen, ohne Anfrage")

print("Einstellungsdialog:")
from otr_review.settings_dialog import SettingsDialog                    # noqa: E402
dlg = SettingsDialog(iface.mainWindow(), core.clean_settings(panel.settings))
vals = core.clean_settings({**panel.settings, **dlg.values()})
base = core.clean_settings(panel.settings)
print("   Unterschiede:", {k: (base[k], vals[k]) for k in base if base[k] != vals[k]})
check(vals == base, "Dialog gibt unveränderte Einstellungen zurück")

print("Bild des Standorts:")
from otr_review.render import image_png, render_selected                  # noqa: E402
iface.canvas.setLayers([layer])
img = render_selected(iface.canvas, layer, [f for f in layer.getFeatures()][2], panel.settings, 200)
png = image_png(img)
open(os.path.join(tmp, "site.png"), "wb").write(png)
check(len(png) > 1000 and img.width() >= 256, f"Standort gerendert ({img.width()}x{img.height()}), {tmp}/site.png")
print("   BILD:", os.path.join(tmp, "site.png"))

print("Einstellungen:")
panel.store_settings(core.clean_settings({**panel.settings, "max_tokens": 1234, "prefetch": 2}))
check(core.load_settings(__import__("qgis.core", fromlist=["QgsSettings"]).QgsSettings().value(core.PREFS_KEY))["max_tokens"] == 1234,
      "Einstellungen gespeichert und wieder lesbar")

panel.close()
plugin.unload()
check(not iface.menu and not iface.toolbar, "unload räumt Menü und Werkzeugleiste auf")
print("\nALLES OK")
app.exitQgis()
