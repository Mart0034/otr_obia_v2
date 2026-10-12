"""Baut qgis_plugin/dist/otr_review.zip - die Datei für QGIS: Erweiterungen > Aus ZIP installieren."""

import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "otr_review")
OUT = os.path.join(HERE, "dist", "otr_review.zip")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(PKG)):
            if name.endswith((".py", ".txt", ".png")) and name != "__pycache__":
                zf.write(os.path.join(PKG, name), f"otr_review/{name}")
    print(OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
