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

### Nachbarschaft für Segmente mittlerer Konfidenz verlangen (optional)

`--require-neighbor-below` (standardmäßig aus) verwirft ein positiv
vorhergesagtes Segment mit `dump_proba` unter dem angegebenen Wert, wenn
kein räumlich angrenzendes Segment derselben Mine ebenfalls positiv
vorhergesagt wurde. Zielt auf ein wiederkehrendes Muster bei den
Falsch-Positiven: isolierte Einzelsegmente mittlerer Konfidenz entlang von
Fahrstraßen und Gruben-Rändern. Segmente ab dem angegebenen Wert werden nie
verworfen, auch ohne Nachbarn - manche echten Dumps bestehen aus genau
einem Segment (der einzige bekannte Dump bei mine_012 ist ein einzelnes,
hochkonfident erkanntes Segment ohne jeden Nachbarn), ein pauschales
"mindestens 2 Segmente"-Kriterium hätte also genau diesen Treffer gelöscht.
Ein sinnvoller Startwert ist `0.6`, derselbe Schwellwert, den die
Vergleichstabelle ohnehin schon ausgibt.

### Kompakte Cluster-Form für Segmente mittlerer Konfidenz verlangen (optional)

`--require-compact-cluster-below` (standardmäßig aus) verwirft ein positiv
vorhergesagtes Segment mit `dump_proba` unter dem angegebenen Wert, wenn
das zusammenhängende Cluster berührender positiver Segmente, zu dem es
gehört, nicht einigermaßen haufenförmig ist, sondern eine lange, dünne
Kette. Das ist eine andere Prüfung als `--require-neighbor-below`, kein
Ersatz dafür: eine Kette von Segmenten entlang einer Straße oder
Klippenkante berührt sich gegenseitig, jedes Segment darin hat also schon
einen unterstützenden Nachbarn und rutscht unbeschadet durch diesen
Filter. Das Problem einer Kette ist nicht Isolation, sondern Form. Die
Cluster-Form wird genauso gemessen wie `shape_compactness`
(4π·Fläche/Umfang²), nur auf das gesamte zusammenhängende Cluster
angewendet statt auf ein einzelnes Segment - bleibt dadurch auch bei
einer kurvigen Straße zuverlässig, anders als ein Seitenverhältnis-Test
der Bounding Box, dem eine gewundene Kette durch eine eher quadratische
Bounding Box entgehen kann. `--min-cluster-compactness` (Standard `0.15`)
legt den Schwellwert fest; an den echten Daten kalibriert, wo
mehrsegmentige Falsch-Positiv-Cluster eine mediane Kompaktheit von 0,27
hatten (20% lagen unter 0,15) gegenüber 0,34 bei echten mehrsegmentigen
Dump-Clustern (0% lagen unter 0,15). Segmente ab dem Konfidenz-Wert
werden nie verworfen, gleiche Begründung wie beim Nachbarschafts-Filter -
hier aber ein höherer Wert als bei `--require-neighbor-below` sinnvoll:
dass ein einzelnes hochkonfidentes Segment ein echter Dump ist, ist
plausibel (mine_012), dass eine ganze Kette aus 20+ Segmenten, bei der
sich das Modell durchgängig sicher ist, ein echter Dump ist, fast nie -
echte Dumps sehen unabhängig von der Modell-Konfidenz nicht wie Straßen
aus. An echten Daten getestet: eine Straßen-Kette in mine_043, bei der
jedes Segment zwischen 0,60 und 0,71 Konfidenz lag, rutschte bei einem
Schwellwert von 0,6 unbeschadet durch; mit 0,8 wurde sie erfasst. `0.6`
für `--require-neighbor-below` und `0.8` für
`--require-compact-cluster-below` ist ein sinnvoller Startwert.

### Sentinel-1-Radar-Merkmale hinzufügen (optional)

Radar misst die Oberflächenrauheit statt der Farbe: ein Reifenhaufen
streut das Signal stark, eine planierte Fahrstraße kaum. Genau diese
Verwechslung steckt hinter den meisten optischen Falsch-Positiven. Außerdem
spielen Wolken keine Rolle.

1. Passende Radardaten zu jedem Minenbild herunterladen (kostenlos, ohne
   Account, von Microsoft Planetary Computer):

   ```bash
   python src/fetch_sentinel1.py --imagery-dir data/imagery --out-dir data/sentinel1 \
     --start 2023-01-01 --end 2023-12-31
   ```

   Pro Mine wird der Median aus bis zu 12 Radaraufnahmen des Zeitraums
   gebildet (`--max-scenes`), was das Radar-Rauschen ("Speckle") unterdrückt,
   und exakt auf das Pixelraster des Sentinel-2-Bilds der Mine gebracht.
   Bereits geladene Minen werden übersprungen, ein abgebrochener Lauf lässt
   sich also einfach neu starten. Wenn bekannt, denselben Zeitraum wie das
   Sentinel-2-Komposit wählen.

2. Pipeline mit `--s1-dir data/sentinel1` starten. Die Segmentierung nutzt
   weiterhin nur Sentinel-2; Radar liefert sechs zusätzliche Merkmale pro
   Segment (Mittelwert und Streuung von VV, VH und dem VH/VV-Verhältnis).
   Eine Mine ohne Radardatei bekommt eine Warnung und leere Radar-Merkmale.
   Findet `--use-cache` einen alten Zwischenspeicher ohne Radar-Merkmale,
   wird er neu berechnet.

### Gelände-Merkmale aus dem Copernicus-Höhenmodell hinzufügen (optional)

Beschattete Berghänge sehen im optischen Bild dunkel und unruhig aus, ähnlich
wie eine Reifenhalde. Gelände-Merkmale helfen dem Modell, einen steilen,
beschatteten Hang von dem meist flachen Untergrund einer Halde zu
unterscheiden.

1. Höhenmodell herunterladen (Copernicus GLO-30, 30m, kostenlos, ohne
   Account), direkt auf das Pixelraster jeder Mine gebracht:

   ```bash
   python src/fetch_dem.py --imagery-dir data/imagery --out-dir data/dem
   ```

2. Pipeline mit `--dem-dir data/dem` starten (kombinierbar mit `--s1-dir`).
   Pro Segment kommen Mittelwert und Streuung dieser Werte hinzu:
   - `dem_slope`: Hangneigung in Grad
   - `dem_tpi`: Höhe minus Mittel der Umgebung (~300m,
     `dem_tpi_window_px`); positiv auf Graten, negativ in Tälern
   - `dem_northness`: -1..1, positiv an nach Norden geneigten Hängen. Auf
     der Südhalbkugel steht die Sonne im Norden, nach Süden geneigte Hänge
     (negativ) liegen also im Schatten. Flaches Gelände = 0.
   - `dem_hillshade_dec` / `dem_hillshade_jun`: erwartete Beleuchtungsstärke
     (-1..1, ungefähr) aus dem tatsächlichen Sonnenstand am lokalen Mittag
     zur Süd-Sommer- bzw. Süd-Wintersonnenwende, berechnet mit einer echten
     Sonnenpositions-Berechnung statt eines reinen Nord/Süd-Näherungswerts.
     Genauer als `dem_northness` in einem engen Tal, wo ein Segment sowohl
     eine besonnte als auch eine beschattete Talwand umfassen kann und der
     northness-*Mittelwert* sich gegenseitig aufhebt, obwohl die Hälfte des
     Segments tatsächlich dunkel liegt. Erfasst nur die Ausrichtung einer
     Fläche zur Sonne, keinen Schattenwurf durch umliegendes Gelände (dafür
     wäre Raytracing über das gesamte Höhenmodell nötig).

   Die absolute Höhe ist bewusst nicht dabei: sie reicht je nach Mine von
   der Küste bis ~4000m und würde vor allem verraten, um welche Mine es
   sich handelt.

### Sentinel-2-Zeitreihen-Merkmale hinzufügen (optional)

Schatten wandern mit der Sonne, Reifenhalden nicht. Über Antofagasta steht
die Sonne beim Sentinel-2-Überflug im Dezember ~23° vom Zenit, im Juni ~56°.
Ein Hang oder eine Grubenkante, die im Winter im Schatten liegt, ist im
Sommer besonnt, während eine Halde das ganze Jahr ungefähr gleich dunkel
bleibt.

1. Helligkeits-Statistik pro Pixel aus bis zu 8 wolkenarmen
   Sentinel-2-Aufnahmen übers Jahr berechnen (kostenlos, ohne Account):

   ```bash
   python src/fetch_s2_timeseries.py --imagery-dir data/imagery --out-dir data/s2_temporal \
     --start 2023-01-01 --end 2023-12-31
   ```

   Wolken und Wolkenschatten werden über die Szenenklassifikation
   ausmaskiert, Geländeschatten bleibt bewusst drin. Ein ganzes Jahr wählen,
   damit Sommer- und Wintersonnenstand dabei sind.

2. Pipeline mit `--s2t-dir data/s2_temporal` starten (kombinierbar mit
   `--s1-dir` und `--dem-dir`). Pro Segment kommen Mittelwert und Streuung
   dieser Werte hinzu:
   - `s2t_bright_cv`: wie stark die Helligkeit übers Jahr schwankt (0 = konstant)
   - `s2t_bright_min_ratio`: dunkelste Aufnahme geteilt durch die typische
     (Median-)Helligkeit; klein = die Stelle liegt zeitweise stark im Schatten

### Sentinel-1-Zeitreihen-Merkmale hinzufügen (optional)

Radar-Rückstreuung hängt stark vom Blickwinkel ab. Eine geometrisch klare
Fläche (Straße, Gebäudekante, Grubenwand) streut je nach Blickrichtung
unterschiedlich stark zurück, ihre Rückstreuung schwankt also zwischen
Aufnahmen. Ein ungeordneter Reifenhaufen streut diffus in die meisten
Richtungen und bleibt vergleichsweise konstant. Dieselbe Idee wie beim
Sentinel-2-Zeitreihen-Merkmal oben, nur Blickwinkel statt Sonnenstand.

1. Zeitliche Schwankung pro Pixel aus bis zu 12 Sentinel-1-RTC-Aufnahmen
   übers Jahr berechnen (kostenlos, kein Account nötig):

   ```bash
   python src/fetch_sentinel1_timeseries.py --imagery-dir data/imagery --out-dir data/s1_temporal \
     --start 2023-01-01 --end 2023-12-31
   ```

2. Pipeline mit `--s1t-dir data/s1_temporal` starten (kombinierbar mit
   `--s1-dir`, `--dem-dir`, `--s2t-dir`). Pro Segment kommen Mittelwert und
   Streuung dieser Werte hinzu:
   - `s1t_vv_cv` / `s1t_vh_cv`: Standardabweichung / Mittelwert der linearen
     Rückstreuung über alle Aufnahmen (0 = bei jeder Aufnahme exakt gleiche
     Rückstreuung, hoch = starke blickwinkelabhängige Schwankung)

   Stichprobenartig an 3 Minen getestet (Escondida/`mine_019`, `mine_043`,
   `mine_079`): keine klare Verbesserung gegenüber den bereits vorhandenen
   räumlichen Textur-Merkmalen `s1_vv_std`/`s1_vh_std` bei dieser kleinen
   Stichprobe - weder geschadet noch klar geholfen. Sollte an mehr Minen
   noch einmal geprüft werden, bevor man sich darauf verlässt.

### Übernommen: räumlicher Helligkeits-Split (Hang-/Schatten-Falsch-Positive)

`spatial_split_contrast` (automatisch berechnet, kein Flag nötig - ein
echtes Modell-Merkmal wie `tex_contrast`, keine optionale OSM-Spalte)
misst die größte Helligkeitsdifferenz zwischen den zwei Hälften der
Bounding Box eines Segments (links/rechts oder oben/unten, je nachdem
was größer ist). Anders als `brightness_extreme_fraction` (zählt nur,
WIE VIEL eines Segments extrem ist, egal WO) erfasst das gezielt einen
sauberen räumlichen Schnitt quer durchs Segment - die Signatur eines
beschatteten Hangs oder einer Grubenkante (eine Seite hell, die andere
dunkel), nicht eines diffus beleuchteten Dumps. Vor der Übernahme an den
echten Daten validiert: `>= 0.10` betrifft 0 von 36 echten Dumps und
fängt ~7% einer Hang-Falsch-Positiv-Stichprobe ab (`dem_slope_mean >= 5`
- dieselbe Klasse, die sonst nichts trennen konnte: Form, DSI-Textur,
Gelände, SAR, NIR/SWIR-Bandverhältnisse, NIR-Band-Textur).

### Untersucht: Gebäude von Dumps trennen (nicht so übernommen)

Gebäude (Lagerhallen, Schuppen) sind eine hartnäckige Falsch-Positiv-
Kategorie - rechteckige Dächer mit scharfer Hell/Schatten-Kante sehen in
mehreren Merkmalen einem Dump ähnlich. Zwei Ideen wurden an den echten 36
gelabelten Dump-Segmenten geprüft, bevor entschieden wurde, ob sie sich
lohnen:

- **`shape_rectangularity` + `brightness_extreme_fraction` stärker
  gewichten** (z.B. als zusätzlicher Abwertungs-Filter wie
  `apply_cluster_shape_filter`) - verworfen. An den echten Daten haben
  echte Dumps und Falsch-Positive fast identische
  Rechteckigkeits-Verteilungen (Median 0.742 vs. 0.738); SLIC-Segmente
  sind durch die Segmentierung selbst schon einigermaßen kastenförmig,
  das Merkmal spiegelt also eher den Segmentierungs-Algorithmus wider als
  das eigentliche Objekt. Jeder Schwellwert, der stark genug ist, um
  Falsch-Positive spürbar zu reduzieren, erwischte auch bis zu 14% der
  echten Dumps (5/36 bei einer lockeren Einstellung); der einzige
  Schwellwert, der keinen einzigen echten Dump erwischte, entfernte nur
  1.8% der grenzwertigen Falsch-Positiven. Nicht der Mühe wert.
- **OSM-Gebäudeumrisse als harter Ausschluss-Filter** - in dieser Form
  verworfen, sicher aber im Schnitt schwach. Keines der 36 echten Dumps
  überlappt ein OSM-Gebäude-Polygon (auch nicht mit einer
  Nachbarschafts-Ausweitung, versucht nach der Idee, dass sich die
  lückenhafte Punktabdeckung so "ausbreiten" ließe - nur 3 zusätzliche
  Segmente von 70.317, also kein wirklicher Effekt in beide Richtungen),
  ein Filter würde also nie eine echte Erkennung kosten. Aber auch nur
  ~1.6% der grenzwertigen Falsch-Positiven (proba 0.5-0.8) überlappen
  eines, gemittelt über alle 138 Minen - die OSM-Gebäudeabdeckung für
  abgelegene Industrieflächen in der Atacama ist extrem ungleichmäßig,
  von null bei den meisten Minen bis ~20-30% der Falsch-Positiven bei
  einer Handvoll gut kartierter Standorte (z.B. der
  "Minera Mechilla"-Aufbereitungsanlage, `mine_043`). Stattdessen als
  informative `is_building`-Spalte statt als Filter umgesetzt - siehe
  "Segmente markieren, die ein OSM-Gebäude überlappen (optional)" weiter
  unten.

  Hinweis zum Overpass-Mirror: `overpass.openstreetmap.fr` liefert an
  Pythons `requests`-Bibliothek gezielt 403 ("only available to
  white-listed usages"), selbst bei identischer Anfrage über denselben
  Proxy, während reines `curl` gegen denselben Mirror anstandslos
  durchgeht - sieht nach TLS-Fingerprint-basierter Bot-Sperre aus, nicht
  nach einem Netzwerk-Policy- oder Header-Problem. Auf einem minimalen
  Runtime-Container ohne installiertes `curl` (z.B. dieses Projekts
  VPS-Deployment, wo per apt installierte Pakete aus dem Install-Schritt
  nicht in den laufenden Container übernommen werden) ist `curl` als
  Workaround aber auch keine Option. `fetch_osm_buildings.py`/
  `fetch_osm_roads.py` nutzen deshalb `overpass-api.de` (die offizielle
  Hauptinstanz) mit reinem `requests` und explizitem `User-Agent`-Header
  - dort gibt es kein 403.

- **OSM-`landuse=industrial`/`quarry`-Polygone als harter Ausschluss** -
  schlimmer als nutzlos, klar verworfen. Jede Mine ist selbst als
  Industrie- oder Abbaufläche in OSM getaggt, und jeder bekannte Dump
  liegt auf Minen-Gelände, also überlappen **auch alle 36 echten Dumps
  eines dieser Polygone, zu 100%** (während es tatsächlich ~92% der
  grenzwertigen Falsch-Positiven erwischt, z.B. den
  "Mina Michilla"-Aufbereitungskomplex). Dieses Signal trennt nur
  "innerhalb einer Mine" von "außerhalb einer Mine", nicht "Dump" von
  "Gebäude" innerhalb einer Mine - nie wieder auf Minen-Grundstücks-Ebene
  testen, das ist der völlig falsche Maßstab.

  Fazit nach drei verworfenen Ideen (Rechteckigkeit/Bimodalität stärker
  gewichten, OSM-Gebäude, OSM-Landuse): kleine Gebäude, die kleiner als
  ein Segment sind (feiner als ~10m gegenüber Sentinel-2), lassen sich
  weder über Form, Textur noch über irgendeine bisher getestete
  kostenlose Zusatz-Vektorebene von Dumps trennen. Das sieht nach einem
  echten Material-Identitäts-Problem aus, nicht nach einem Form-/
  Kontext-Problem - siehe "PRISMA-Hyperspektraldaten" unter Zukünftige
  Arbeit weiter unten für die eine noch nicht getestete Option, die dafür
  tatsächlich geeignet wäre.

### Segmente markieren, die ein OSM-Gebäude überlappen (optional)

1. Gebäudeumrisse pro Mine abrufen (kostenlos, kein Account - siehe die
   Untersuchung oben, warum das informativ bleibt statt ein Filter zu
   werden: die OSM-Abdeckung ist zu ungleichmäßig, um ihr überall
   gleichermaßen zu trauen):

   ```bash
   python src/fetch_osm_buildings.py --imagery-dir data/imagery --out-dir data/osm_buildings
   ```

2. Pipeline mit `--osm-buildings-dir data/osm_buildings` starten. Die
   exportierte Karte bekommt eine zusätzliche Spalte:
   - `is_building`: `True` für ein Segment, das ein kartiertes
     OSM-Gebäude überlappt oder ein solches Segment berührt; sonst
     `False` (auch für jede Mine ohne abgerufene `<mine_id>.gpkg` oder
     mit einer leeren - nie ein Fehler).

   Das ändert **niemals `dump_pred` oder `dump_proba`** - die Spalte ist
   dafür da, diese Segmente selbst in QGIS auszublenden (Rechtsklick auf
   Layer → Filter, oder Eigenschaften → Quelle → Abfrage-Editor:
   `"is_building" = 0`, optional kombiniert mit `"dump_pred" = 1`) an den
   Minen, wo die OSM-Abdeckung gut genug ist, statt dass die Pipeline das
   für alle Minen auf einmal entscheidet.

### Segmente in einem dichten Straßen-Gitter markieren (optional)

Bloße Nähe zu einer Straße trennt kaum echte Dumps von Falsch-Positiven -
11% der 36 echten Dumps liegen innerhalb von 12m einer kartierten Straße,
da ein einzelner Fahrzeug-Zufahrtsweg zu einem echten Dump völlig normal
ist. Ein *dichtes Gitter* aus vielen, meist kurzen/parallelen/
rechtwinkligen Straßen ist dagegen die Signatur eines Parkplatz- oder
Fahrzeugbereichs einer Industrieanlage, nicht eines einzelnen Dumps -
kalibriert bei `road_density_150m >= 500` (Gesamtlänge kartierter
Straßen innerhalb 150m eines Segments): 0 von 36 echten Dumps betroffen,
~6.5% der grenzwertigen Falsch-Positiven (proba 0.5-0.8) erfasst, besser
als das Gebäude-Überlappungs-Signal oben.

1. Straßen pro Mine abrufen (kostenlos, kein Account):

   ```bash
   python src/fetch_osm_roads.py --imagery-dir data/imagery --out-dir data/osm_roads
   ```

2. Pipeline mit `--osm-roads-dir data/osm_roads` starten. Die exportierte
   Karte bekommt zwei zusätzliche Spalten:
   - `road_density_150m`: Gesamtlänge (Meter) kartierter OSM-Straßen
     innerhalb 150m um den Segment-Mittelpunkt (`--road-density-radius-m`
     ändert den Radius)
   - `is_road_grid`: `road_density_150m >= 500`
     (`--road-grid-threshold-m` ändert den Schwellwert - niedrigere Werte
     erfassen mehr Falsch-Positive, kosten aber ab ~300m echte Dumps,
     siehe die Untersuchung oben)

   Wie bei `is_building` ändert das **niemals `dump_pred`/`dump_proba`**
   - selbst in QGIS ausblenden (`"is_road_grid" = 0`), wo es passt.

### Eine völlig neue Stelle prüfen (ohne vorhandene Kachel)

Alle 138 vorhandenen `data/imagery/*.tif`-Kacheln stammen aus "Stage 2"
der früheren Machbarkeitsstudie, nicht aus diesem Repository (siehe oben)
- bisher gab es also keine Möglichkeit, die Pipeline auf eine Stelle
anzusetzen, die diesen Prozess noch nicht durchlaufen hat.
`fetch_s2_base_imagery.py` baut denselben 10-Band-Stack von Grund auf für
einen beliebigen Mittelpunkt + Radius:

```bash
python src/fetch_s2_base_imagery.py --center-x 365950.1 --center-y 7367190.3 \
  --radius-m 1500 --mine-id new_001 --out-dir data/new_sites/imagery
```

`blue/green/red/nir/swir1/swir2` sind normale Sentinel-2-L2A-Reflektanz,
und `ndvi`/`ndwi`/`bsi` wurden anhand der echten `data/imagery`-Kacheln
exakt zurückgerechnet (Fehler < 1e-7):

```
ndvi = (nir - red) / (nir + red)
ndwi = (green - swir1) / (green + swir1)          # ein modifiziertes NDWI, nicht die übliche (green-nir)-Version
bsi  = ((swir1+red) - (nir+blue)) / ((swir1+red) + (nir+blue))
```

**`dsi` ist `-(red + nir + swir1) / 3`.** Zunächst wirkte es
nicht rekonstruierbar, weil jeder verhältnisartige Index scheiterte. Eine
lineare Regression des DSI-Bands der echten Kacheln auf die 6 Rohbänder
liefert R^2 = 1.000 mit Koeffizienten von genau -1/3 für red, nir und swir1
und 0 für den Rest (max. Fehler 4e-8, float32-Genauigkeit, auch an nicht
in die Anpassung eingegangenen Kacheln bestätigt). Kacheln neuer Stellen
haben damit dieselben `dsi_mean`/`dsi_std`- und GLCM-Textur-Merkmale wie die
138 bekannten Minen und sind direkt mit ihnen vergleichbar.

Um eine neue Stelle tatsächlich zu prüfen: `--s1-dir`/`--dem-dir`/`--s2t-dir`
wie gewohnt für die neue Kachel abrufen, dann die Pipeline mit deren
Bildordner zusammen mit den 138 bekannten Minen laufen lassen (damit das
Modell weiterhin an echten, gelabelten Dumps trainiert und nur die
ungelabelten Segmente der neuen Stelle zusätzlich bewertet) statt allein -
eine einzelne ungelabelte Kachel liefert keine Trainingsdaten für einen
Klassifikator.

Um *nur* die neue Kachel zu bewerten, ohne die 138 Minen neu zu segmentieren,
lässt sich der Zwischenspeicher eines früheren Volllaufs wiederverwenden:
Ausgabeordner kopieren, die neue(n) Kachel(n) in einen eigenen Bilderordner
legen und `--use-cache --extra-imagery-dir <Ordner>` angeben. Die Kachel wird
segmentiert und an die zwischengespeicherten Daten angehängt (trainiert wird
weiter auf allen Minen), exportiert werden nur ihre Segmente:

```bash
cp -r output_full_138 output_new_002
python src/otr_obia_pipeline.py ... --output-dir output_new_002 --use-cache \
  --extra-imagery-dir data/new_only/imagery
```

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
- Kleine Gebäude (kleiner als ein Segment, also feiner als ~10m) bleiben
  eine hartnäckige Falsch-Positiv-Quelle, die aktuell nichts in der
  Pipeline zuverlässig von echten Dumps trennt - siehe "Untersucht:
  Gebäude von Dumps trennen" oben für das, was ausprobiert und verworfen
  wurde.

## Zukünftige Arbeit

- **PRISMA-Hyperspektraldaten** (ASI, ~200+ schmale Spektralbänder, 30m
  Auflösung, kostenlos mit Registrierung, bestätigte Abdeckung über
  Chile) ist die vielversprechendste noch nicht getestete Option für das
  Problem "kleine Gebäude vs. Dumps" oben. Jede bisher getestete Idee
  (Form, Textur, OSM-Gebäude, OSM-Landuse) arbeitet mit groben
  Sentinel-2-Spektralbändern oder Zusatz-Vektordaten, keines davon kann
  Gummi von Dach-/Baumaterial oder Gestein auf chemischer Ebene
  unterscheiden. PRISMAs feine Spektralauflösung ist genau für diese Art
  Materialerkennung gebaut und wurde noch nicht ausprobiert.
- **ALOS PALSAR** (JAXA L-Band-SAR, kostenlos, 25m, verfügbar über
  AWS/GEE) als zusätzliche Radarquelle neben dem bereits vorhandenen
  Sentinel-1-C-Band - niedrigere Priorität als PRISMA, da Sentinel-1
  bereits die Frage "streut das wie ein ungeordneter, diffuser Haufen"
  bedient; L-Band wäre eine Variation eines bereits genutzten Signals,
  keine neue Achse.
