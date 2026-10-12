"""Fenster 'Beispielbilder verwalten': Liste mit Vorschau, Beschriftung ändern, an/aus, löschen."""

from .qtcompat import (QCheckBox, QDialog, QHBoxLayout, QIcon, QInputDialog, QLabel, QPixmap, QPushButton, QSize,
                       QTableWidget, QTableWidgetItem, QVBoxLayout, qt)


class ExamplesDialog(QDialog):
    def __init__(self, parent, store):
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("Beispielbilder")
        self.resize(760, 520)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Bild", "Art", "Beschriftung", "aktiv"])
        self.table.setIconSize(QSize(120, 120))
        self.table.verticalHeader().setDefaultSectionSize(124)
        self.table.setColumnWidth(0, 128)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(2, 380)
        self.table.setColumnWidth(3, 50)
        info = QLabel("Diese Bilder (von dir bestätigt) gehen mit jeder KI-Anfrage vor dem neuen Bild mit. "
                      "Wenige, gute Beispiele sind besser als viele; ausgewogen: Hälfte Dump, Hälfte Kein Dump.")
        info.setWordWrap(True)
        rename = QPushButton("Beschriftung ändern")
        rename.clicked.connect(lambda _=False: self.rename())
        delete = QPushButton("Löschen")
        delete.clicked.connect(lambda _=False: self.delete())
        close = QPushButton("Schließen")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        for b in (rename, delete):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(self.table, 1)
        layout.addLayout(row)
        self.reload()

    def reload(self):
        entries = self.store.list()
        self.table.setRowCount(len(entries))
        for r, m in enumerate(entries):
            pm = QPixmap()
            pm.loadFromData(self.store.png_bytes(m["id"]))
            item = QTableWidgetItem()
            item.setIcon(QIcon(pm))
            item.setData(qt("ItemDataRole", "UserRole"), m["id"])
            self.table.setItem(r, 0, item)
            self.table.setItem(r, 1, QTableWidgetItem("Dump" if m["kind"] == "dump" else "Kein Dump"))
            self.table.setItem(r, 2, QTableWidgetItem(m["caption"]))
            box = QCheckBox()
            box.setChecked(m.get("enabled", True))
            box.toggled.connect(lambda on, i=m["id"]: self.store.set_enabled(i, on))
            self.table.setCellWidget(r, 3, box)

    def _current_id(self):
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.data(qt("ItemDataRole", "UserRole")) if item is not None else None

    def rename(self):
        ex_id = self._current_id()
        if ex_id is None:
            return
        row = self.table.currentRow()
        text, ok = QInputDialog.getText(self, "Beschriftung", "Kurze Beschreibung für die KI:",
                                        text=self.table.item(row, 2).text())
        if ok:
            self.store.set_caption(ex_id, text)
            self.reload()

    def delete(self):
        ex_id = self._current_id()
        if ex_id is not None:
            self.store.delete(ex_id)
            self.reload()
