"""Qt5/Qt6-Unterschiede an einer Stelle (QGIS 3 = Qt5, QGIS 4 = Qt6)."""

from qgis.PyQt import QtCore, QtGui, QtWidgets
from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QSize, Qt, QTimer, QUrl
from qgis.PyQt.QtGui import QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygonF
from qgis.PyQt.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
                                 QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QInputDialog, QLabel,
                                 QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                                 QSpinBox, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

try:                         # Qt6: QAction und QShortcut liegen in QtGui, Qt5: in QtWidgets
    from qgis.PyQt.QtGui import QAction
except ImportError:
    from qgis.PyQt.QtWidgets import QAction
try:
    from qgis.PyQt.QtGui import QShortcut
except ImportError:
    from qgis.PyQt.QtWidgets import QShortcut


def enum(cls, group, name):
    """Qt6 kennt nur cls.Gruppe.Name, Qt5 (meist) nur cls.Name - beides unterstützen."""
    grp = getattr(cls, group, None)
    if grp is not None and hasattr(grp, name):
        return getattr(grp, name)
    return getattr(cls, name)


def qt(group, name):
    return enum(Qt, group, name)


def make_field(name, kind):
    """QgsField für 'string'/'double': QGIS 4 (Qt6) will QMetaType.Type, QGIS 3 (Qt5) QVariant.Type."""
    from qgis.core import QgsField
    last = None
    for build in (_metatype, _variant):
        try:
            return QgsField(name, build(kind))
        except (TypeError, ImportError, AttributeError) as exc:
            last = exc
    raise last


def _metatype(kind):
    from qgis.PyQt.QtCore import QMetaType
    return getattr(QMetaType.Type, "QString" if kind == "string" else "Double")


def _variant(kind):
    from qgis.PyQt.QtCore import QVariant
    return QVariant.String if kind == "string" else QVariant.Double


def standard_buttons(*names):
    box = QDialogButtonBox
    out = None
    for n in names:
        v = enum(box, "StandardButton", n)
        out = v if out is None else out | v
    return out


def exec_dialog(dlg):
    """dlg.exec() (Qt6) / exec_() (ältere Qt5-Bindungen)."""
    return dlg.exec() if hasattr(dlg, "exec") else dlg.exec_()
