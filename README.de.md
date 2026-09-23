*In anderen Sprachen lesen: [English](README.md) | **Deutsch***

# OTR-OBIA-Pipeline

Erkennt Reifenhalden ("OTR" steht für Old Tire Dump) an Minenstandorten
anhand von Sentinel-2-Satellitenbildern, mit einem objektbasierten
Ansatz (OBIA): Jedes Bild wird zunächst in homogene Flächen ("Segmente")
zerlegt, und jedes Segment wird als Ganzes mit einem Random Forest
klassifiziert, statt jedes einzelne Pixel mit einem CNN zu
klassifizieren. Den vollständigen Hintergrund und die Begründung findet
man im Docstring des Moduls in `src/otr_obia_pipeline.py`.

## Status

- Pipeline-Code: implementiert und von Bugs bereinigt.
- Automatisierte Tests: vorhanden (`tests/`), laufen auf künstlichen
  Testdaten.
- Echte Daten: **noch nicht vorhanden**. Bisher lief die Pipeline nur
  gegen künstliche Testdaten, noch nie gegen echte Sentinel-2-Bilder.
  Siehe "Benötigte Daten" weiter unten.

## Einrichtung

Erfordert Python 3.10+. Eine eigene virtuelle Umgebung wird empfohlen,
da einige dieser Bibliotheken (rasterio, geopandas) mit dem in QGIS
mitgelieferten Python in Konflikt geraten können. Die QGIS-Python-Umgebung
sollte daher nicht mitverwendet werden.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # nur zum Ausführen der Pipeline
# oder, um zusätzlich die Tests auszuführen:
pip install -r requirements-dev.txt
```

## Benötigte Daten

Die Pipeline bringt keine eigenen Daten mit. Zwei Dinge müssen unter
`data/` abgelegt werden (dieser Ordner ist in `.gitignore` eingetragen,
es landet also nichts davon im Git-Repository):

1. **`data/imagery/*.tif`**: ein GeoTIFF pro Mine, jeweils mit genau
   10 Bändern, in dieser Reihenfolge:

   ```
   blue, green, red, nir, swir1, swir2, ndvi, ndwi, dsi, bsi
   ```

   Diese sollten bereits aus "Stage 2" des früheren
   Feasibility-Study-Projekts vorliegen. Am besten bei der Person
   nachfragen, die diese Stage durchgeführt hat, wo die fertigen
   Bildstapel pro Mine liegen.

2. **`data/dump_labels.gpkg`**: ein GeoPackage (oder Shapefile) mit den
   in QGIS digitalisierten Reifenhalden-Polygonen, mit einer Spalte
   (Standardname: `mine_id`), die angibt, zu welcher Mine jedes Polygon
   gehört.

3. **Die Minen-IDs müssen übereinstimmen.** Die Pipeline versucht, die
   Minen-ID aus dem Dateinamen des Bildes zu erraten (z. B. wird aus
   `mine_12.tif` die ID `mine_12`, aus `7.tif` die ID `7`). Diese
   erratene ID muss zu den Werten in der `mine_id`-Spalte der
   Label-Datei passen, sonst werden alle Segmente dieser Mine
   stillschweigend als "kein Dump" markiert. Falls das passiert, gibt
   es dafür eine klare Warnung im Log, aber es lohnt sich, das
   Namensschema vorher zu prüfen (siehe `CONFIG["mine_id_field"]` /
   `extract_mine_id()` im Skript).

4. **`data/mine_boundaries.gpkg`** (optional, aber empfohlen): die
   tatsächlichen Minen-Grenzpolygone (nicht der gepufferte Bildausschnitt).
   Jeder Sentinel-2-Export enthält einen 500m-Puffer umliegendes Gelände
   rund um die Mine, das nie eine Reifenhalde enthalten kann, aber trotzdem
   segmentiert und klassifiziert wird. Mit
   `--mine-boundary-path data/mine_boundaries.gpkg` (Spalte `mine_id` als
   Standard, siehe `--mine-boundary-id-field`) werden Segmente, die
   komplett außerhalb der echten Grenze liegen, schon vor der
   Klassifikation ausgeschlossen - reduziert False Positives ohne
   Mehraufwand. Ohne diese Option verhält sich die Pipeline exakt wie
   zuvor (kein Filtern).

## Daten vor dem Lauf prüfen

Vor dem eigentlichen (unter Umständen langwierigen) Pipeline-Lauf lohnt
sich der Daten-Check. Er meldet innerhalb weniger Sekunden, ob die
GeoTIFFs die richtige Anzahl Bänder haben, wie viele Pixel fehlen/
NoData sind, und ob die Minen-ID jeder Datei wirklich zu einem Eintrag
in der Label-Datei passt:

```bash
python3 src/check_data.py --imagery-dir data/imagery --labels-path data/dump_labels.gpkg
```

Bei Problemen ist der Exit-Code ungleich null. Eine Mine mit
"KEIN LABEL-MATCH" bedeutet: die aus dem Dateinamen erratene Minen-ID
passt zu keinem Eintrag in der Label-Datei. Das sollte vor dem
eigentlichen Pipeline-Lauf behoben werden, nicht danach.

## Pipeline ausführen

Einstellungen müssen nicht mehr im Code geändert werden. Entweder das
`CONFIG`-Dictionary am Anfang von `src/otr_obia_pipeline.py` anpassen,
oder Optionen direkt auf der Kommandozeile übergeben:

```bash
python3 src/otr_obia_pipeline.py \
  --imagery-dir data/imagery \
  --labels-path data/dump_labels.gpkg \
  --output-dir output
```

`python3 src/otr_obia_pipeline.py --help` zeigt alle verfügbaren
Optionen (dieselben Optionen funktionieren auch bei `check_data.py`).
Eine ganze Reihe von Einstellungen lässt sich auch in einer JSON-Datei
sammeln und mit `--config` übergeben:

```json
{"imagery_dir": "data/imagery", "min_overlap_ratio": 0.4, "n_estimators": 600}
```

```bash
python3 src/otr_obia_pipeline.py --config my_settings.json
```

Einzelne Flags gewinnen immer gegenüber der Config-Datei, die Config-
Datei gewinnt immer gegenüber den eingebauten Standardwerten in
`CONFIG`.

Der Pipeline-Lauf (mit beiden Methoden) macht dann Folgendes:
1. Baut den Segment-Datensatz aus den Bildern und Labels aller Minen auf.
2. Trainiert einen Random Forest und validiert ihn per Kreuzvalidierung
   (gruppiert nach Mine, damit keine Mine gleichzeitig in Training und
   Validierung landet).
3. Klassifiziert alle Segmente und exportiert
   `output/segments_classified.gpkg`.

Dieses GeoPackage kann per Drag & Drop in QGIS geladen werden; zur
Kontrolle nach der Spalte `dump_proba` einfärben (Graduated,
Schwellwert standardmäßig ca. 0,5, siehe unten).

Alles, was die Pipeline tut, wird mit Zeitstempel in die Konsole
geloggt, inklusive Warnungen bei allem, was auffällig aussieht
(nicht zuordenbare Minen-IDs, gefundene No-Data-Pixel, eine
Kreuzvalidierungs-Fold mit nur einer Klasse usw.). Nach jedem Lauf
lohnt sich ein Blick auf die Warnungen im Log.

### Schwellwert für die Klassifikation wählen

`--classification-threshold` (Standard `0.5`) legt fest, ab welchem
`dump_proba`-Wert ein Segment als positiv gilt - sowohl für die
Kreuzvalidierungs-Metriken als auch für die `dump_pred`-Spalte im
exportierten GeoPackage. Bei so unausgeglichenen Daten ist 0,5 nicht
unbedingt der beste Kompromiss: ein höherer Schwellwert (z. B. `0.7`
oder `0.8`) senkt die Anzahl falsch-positiver Segmente deutlich, auf
Kosten davon, ein paar der schwächeren echten Dumps zu verpassen. Jeder
Lauf loggt jetzt automatisch eine Schwellwert-Vergleichstabelle
(Out-of-Fold, also ehrlich), die zeigt, wie viele Minen bei welchem
Schwellwert erkannt werden und wie viele Segmente jeweils als positiv
markiert werden - damit lässt sich ein Wert auswählen und mit
`--use-cache` (überspringt die erneute Segmentierung, trainiert/exportiert
nur neu) erneut laufen lassen, statt zu raten.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest -v
```

Die Tests verwenden kleine, künstlich erzeugte "Minen"-Bilder und
Labels (siehe `conftest.py`). Sie brauchen keine echten Daten und
laufen in wenigen Sekunden durch. Sie laufen außerdem automatisch bei
jedem Push über GitHub Actions (`.github/workflows/tests.yml`).

## Bekannte Einschränkungen

- Die GLCM-Textur-Merkmale werden auf der Bounding Box jedes Segments
  berechnet, nicht auf der exakten Polygonform. Das ist eine einfache
  Näherung, keine exakte Berechnung.
- Die Modell-Einstellungen (`n_segments_per_mine`, `compactness`,
  `min_overlap_ratio`, `n_estimators`) sind erste Standardwerte und
  wurden noch nicht an echten Daten optimiert.
- Bei sehr wenigen Minen kann es vorkommen, dass eine
  Kreuzvalidierungs-Fold nur eine Klasse enthält. Die Pipeline stürzt
  dann nicht ab, aber der resultierende Fold-Score ist wenig
  aussagekräftig. In den geloggten Warnungen steht, welche Minen davon
  betroffen waren.
