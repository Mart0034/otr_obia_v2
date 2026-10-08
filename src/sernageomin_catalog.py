"""
SERNAGEOMIN-ArcGIS-Dienste auflisten
====================================

Listet alle Ordner/Dienste/Layer des SERNAGEOMIN-ArcGIS-Servers mit
Geometrietyp, Objektanzahl und Feldnamen, damit man sieht, welche Daten
(Bergwerke, Halden, Absetzbecken, ...) abrufbar sind. Nur lesend.

Nutzung:
    python src/sernageomin_catalog.py > sernageomin_katalog.txt
"""

import sys

import requests

BASE = "https://sdngsig.sernageomin.cl/gissdng/rest/services"
HEADERS = {"User-Agent": "otr-obia-pipeline/1.0 (research)"}


def get(url, **params):
    r = requests.get(url, params={"f": "json", **params}, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def walk(folder=""):
    info = get(f"{BASE}/{folder}".rstrip("/"))
    for svc in info.get("services", []):
        yield svc["name"], svc["type"]
    for sub in info.get("folders", []):
        yield from walk(f"{folder}/{sub}".lstrip("/") if folder else sub)


def describe(name, kind):
    print(f"\n== {name} [{kind}]")
    if kind not in ("MapServer", "FeatureServer"):
        return
    try:
        layers = get(f"{BASE}/{name}/{kind}").get("layers", [])
    except Exception as exc:
        print(f"   (nicht lesbar: {exc})")
        return
    for lay in layers:
        try:
            meta = get(f"{BASE}/{name}/{kind}/{lay['id']}")
            count = get(f"{BASE}/{name}/{kind}/{lay['id']}/query",
                        where="1=1", returnCountOnly="true").get("count", "?")
            fields = ", ".join(f["name"] for f in meta.get("fields", [])[:12])
            print(f"   [{lay['id']}] {lay['name']}  ({meta.get('geometryType', '-')}, {count} Objekte)")
            print(f"       Felder: {fields}")
        except Exception as exc:
            print(f"   [{lay['id']}] {lay['name']}  (Details nicht lesbar: {exc})")


def main():
    for name, kind in walk():
        describe(name, kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
