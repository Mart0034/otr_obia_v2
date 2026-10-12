"""OTR Review - ein QGIS-Plugin für Kandidatenlisten, Review und KI-Einschätzung."""


def classFactory(iface):
    from .plugin import OtrReviewPlugin
    return OtrReviewPlugin(iface)
