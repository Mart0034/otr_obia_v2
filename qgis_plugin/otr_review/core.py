"""
Kernlogik des Plugins OTR Review (ohne QGIS, daher testbar)
===========================================================

Enthält alles, was nicht von Qt/QGIS abhängt: Einstellungen, Prompt, Anfrage/Antwort an die
Anthropic-API, Preise, Umkreis nach Umgebung, Urteil-Abbildung, Auto-Clean-Regeln, Stichprobe und
die Reihenfolge-Logik der Review-Warteschlange. Die Oberfläche steht in ui.py.
"""

from collections import deque

import base64
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-haiku-5-5"
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"]
MAX_EDGE_PX = 1568          # größere Bilder verkleinert die API ohnehin
BUFFER_M = 50.0             # Umkreis um den Standort, der mitbeurteilt wird
MIN_SIDE_M = 150.0          # kleinster Ausschnitt (Meter), damit winzige Standorte sichtbar bleiben
LOG_PATH = os.path.join(os.path.expanduser("~"), "ai_classifier_log.csv")
PREFS_KEY = "otr_review/prefs"                    # QGIS-Einstellungen (JSON)
OLD_PREFS_KEY = "otr_obia/ai_classifier_prefs"    # Einstellungen der früheren Skripte werden übernommen
KEY_SETTING = "otr_obia/anthropic_api_key"        # derselbe Schlüssel wie früher: nichts neu eintippen

LABELS = ("dump", "clean_road", "clean_desert", "clean_town", "clean_mine", "clean_yard",
          "clean_snow", "clean_water", "clean_vegetation", "clean_other", "unsure")

SYSTEM_PROMPT = """You help review candidate sites for a detection project that looks for \
dumps of discarded giant mining tires (OTR tires) in northern Chile, in satellite imagery.

You are shown a screenshot of a GIS map. A yellow/highlighted polygon, if present, \
marks the candidate site under review. A dashed cyan line, if present, marks the zone \
around the polygon (its radius is given in the message). Judge everything inside the dashed line: the polygon \
itself and its surroundings. Ignore what lies outside the dashed line.

What a tire dump usually looks like: a compact area of dark, speckled or ring-like texture \
(individual tires are 2-4 m across), often piles or neat rows, on bare ground near a mine, \
frequently beside a haul road or waste-rock area. Confusers: roads and haul roads, rock/waste \
piles and pit shadows, equipment yards and workshops, towns and buildings, snow patches, \
dark rock or desert colour variation, water, vegetation.

Tire storage yards and stockpiles count as dumps: stacks or rows of tires, even neat ones, \
on bare ground or inside an equipment yard. Pipes, containers, vehicles and buildings do not. \
The project prefers false alarms to missed dumps: if round objects of tire size that could be tires \
are visible anywhere in the zone, dump_probability must be at least 0.4, and dump_location says where.

Scale decides: the message gives metres per pixel (and a scale bar is drawn in the image). A mining \
tire is only 2.5-4 m across. Tires are uniform hollow rings (black, dark grey or light grey), with \
sharp edges and a lighter or empty centre, lying in rows, stacks or tight piles; their shadows are short.
NOT tires: dirt and rock heaps (spoil heaps, waste-dump slopes, pit benches) - irregular rounded \
mounds typically 8-30 m across with soft shading, no hollow centre and uneven sizes, often in \
scalloped or terraced patterns; the dark shadow side of such mounds can look like dark rings. \
Never call a view a dump only because mounds are dark or speckled; require ring shapes of tire size.

Location is irrelevant: tire dumps are common inside mining operations, next to haul roads, pits \
and terraces. Never lower the probability because the view shows haul roads, terraces or yards.

Classify the view with exactly one label from: %s.
Use "dump" only if tires are plausibly visible; use "unsure" if the image resolution is too \
coarse or the view is ambiguous. Also say where the dump (if any) is: "inside" the highlighted \
polygon, "partial" (polygon only partly covers it), "nearby" (inside the dashed zone but outside \
the polygon), or "none".

Reply with ONLY a JSON object:
{"label": "<label>", "confidence": <0..1, how sure you are of the label>, \
"dump_probability": <0..1, probability that a tire dump is present in the zone>, \
"dump_location": "inside|partial|nearby|none", "reason": "<max 20 words>"}

The reason must be telegraphic: only what is visible and why it decides the label, e.g. \
"Open-pit terraces, dark waste-rock slopes, no tire patterns." Never start with filler such as \
"The zone shows", "The image shows", "This area contains" or "I can see".""" % ", ".join(LABELS)


# prompt_id() früherer Standard-Prompts (vor dem Kürzen der Begründung)
LEGACY_PROMPT_IDS = {"1a72a531", "af179aa7", "bb221802", "e05849c1", "495a90d6", "3270d54d"}

CONTEXTS = ("inside_op", "near_op", "open_desert")
EFFORTS = ("", "low", "medium", "high", "xhigh", "max")

DEFAULT_SETTINGS = {
    "model": DEFAULT_MODEL,
    "max_tokens": 2048,           # Thinking zählt mit: zu klein = Antwort bleibt leer
    "effort": "",                 # "" = API-Standard des Modells; low / medium / high / xhigh / max
    "thinking": "default",        # default | off (off = möglichst wenig/kein Thinking)
    "temperature": None,          # None = API-Standard
    "system_prompt": SYSTEM_PROMPT,
    "radius_m": BUFFER_M,         # Umkreis, wenn der Standort keine Spalte `context` hat (oder Auto aus)
    "auto_radius": True,          # Umkreis nach der Spalte `context` (site_context.py) wählen
    "radius_inside": 200.0,       # im Betriebsgelände einer Mine/eines Steinbruchs: Anlagen mit ansehen
    "radius_near": 150.0,         # am Rand eines Betriebs
    "radius_desert": 100.0,       # offene Wüste (9 gezeichnete Halden: 4 davon in der Wüste, bis 194 m neben dem Standort)
    "max_image_px": 1024,         # Kantenlänge des gerenderten Bildes
    "min_view_m": MIN_SIDE_M,
    "draw_ring": True,
    "timeout_s": 90,
    "log_path": LOG_PATH,
    "price_in": 0.0,              # USD je Mio. Eingabe-Token; 0 = Preistabelle des Modells
    "price_out": 0.0,
    "thr_dump": 0.5,              # ab dieser Wahrscheinlichkeit: "Dump found"
    "thr_possible": 0.2,          # ab hier "Possible dump" (Recall-Vorrang), darunter "No dump"
    "prompt_cache": True,         # System-Prompt zwischenspeichern (Cache-Lesen kostet nur 10 % - ab 512 Token)
    "auto_clean": True,           # Stapel: sichere Wüsten-Standorte automatisch als clean eintragen
    "auto_clean_p": 0.10,         # ... wenn die Halden-Wahrscheinlichkeit höchstens so hoch ist
    "auto_clean_contexts": "open_desert",   # ... und die Umgebung dazu passt (Komma-getrennt)
    "spot_check_every": 10,       # jeder n-te Auto-Clean-Kandidat kommt zur Kontrolle in deine Warteschlange
    "batch_n": 500,               # zuletzt benutzte Stapelgröße
    "batch_parallel": 10,         # gleichzeitige Standorte im Stapel (rendern + Anfrage)
    "batch_render_parallel": 4,   # davon gleichzeitig im Hintergrund gerendert (nur render_mode = clean)
    "use_examples": True,         # gespeicherte Beispielbilder (von dir bestätigt) mit jeder Anfrage mitschicken
    "max_examples": 6,            # höchstens so viele (ausgewogen: Hälfte Dump, Hälfte Kein Dump)
    "example_px": 768,            # Kantenlänge der gespeicherten Beispielbilder (kleiner = billiger)
    "tell_context": False,        # der KI die Umgebung (Betrieb/Wüste) im Text nennen (Standard aus, siehe zone_note)
    "highlight_fill": False,      # clean-Bild: Standort nur umranden (Standard) statt gelb füllen - verdeckt die Reifen nicht
    "render_mode": "clean",       # clean: nur Satellitenbild + Umriss/Umkreis (schnell, nichts verdeckt den Boden);
                                  # visible: alle sichtbaren Schichten wie im Kartenfenster
    "batch_budget_usd": 1.0,      # Stapel stoppt, wenn die geschätzten Kosten das erreichen (0 = kein Limit)
    "prefetch": 4,                # so viele Standorte vor dem aktuellen werden schon vorab eingeschätzt (Auto)
    "sort_by_ai": False,          # Warteschlange: Standorte mit KI-Ergebnis zuerst, hohe Wahrscheinlichkeit vorne
    "auto_next": False,           # nach Annehmen/Ablehnen den nächsten Standort automatisch einschätzen
}
_LIMITS = {"radius_inside": (0, 500, float), "radius_near": (0, 500, float), "radius_desert": (0, 500, float),
           "max_tokens": (16, 32000, int), "radius_m": (0, 500, float), "max_image_px": (256, MAX_EDGE_PX, int),
           "min_view_m": (20, 2000, float), "timeout_s": (10, 600, int), "prefetch": (0, 16, int),
           "spot_check_every": (1, 100, int), "max_examples": (0, 12, int), "example_px": (256, 1568, int), "batch_n": (1, 100000, int), "batch_parallel": (1, 24, int), "batch_render_parallel": (1, 12, int),
           "batch_budget_usd": (0, 1000, float), "auto_clean_p": (0, 0.5, float)}


def clean_settings(raw):
    """Gespeicherte/eingegebene Werte -> gültige Einstellungen (fehlende/kaputte Werte = Standard,
    Zahlen werden auf sinnvolle Grenzen gekürzt, leerer Prompt = Standard-Prompt)."""
    raw = raw if isinstance(raw, dict) else {}
    out = dict(DEFAULT_SETTINGS)
    for key, (lo, hi, typ) in _LIMITS.items():
        try:
            out[key] = typ(max(lo, min(hi, float(raw[key]))))
        except (KeyError, TypeError, ValueError):
            pass
    temp = raw.get("temperature")
    if temp is not None:
        try:
            out["temperature"] = max(0.0, min(1.0, float(temp)))
        except (TypeError, ValueError):
            pass
    if isinstance(raw.get("auto_clean_contexts"), str):
        ctx = [c.strip() for c in raw["auto_clean_contexts"].split(",") if c.strip() in CONTEXTS]
        out["auto_clean_contexts"] = ",".join(ctx)
    for key in ("model", "system_prompt", "log_path"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            out[key] = raw[key].strip() if key != "system_prompt" else raw[key]
    if prompt_id(out["system_prompt"]) in LEGACY_PROMPT_IDS:   # früherer Standard-Prompt -> neuer Standard
        out["system_prompt"] = SYSTEM_PROMPT
    for key in ("thr_dump", "thr_possible"):
        try:
            out[key] = max(0.0, min(1.0, float(raw[key])))
        except (KeyError, TypeError, ValueError):
            pass
    out["thr_possible"] = min(out["thr_possible"], out["thr_dump"])
    for key in ("price_in", "price_out"):
        try:
            out[key] = max(0.0, min(1000.0, float(raw[key])))
        except (KeyError, TypeError, ValueError):
            pass
    if raw.get("render_mode") in ("clean", "visible"):
        out["render_mode"] = raw["render_mode"]
    if raw.get("effort") in EFFORTS:
        out["effort"] = raw["effort"]
    if raw.get("thinking") in ("default", "off"):
        out["thinking"] = raw["thinking"]
    for key in ("draw_ring", "auto_next", "auto_radius", "prompt_cache", "auto_clean", "tell_context", "highlight_fill",
                "use_examples", "sort_by_ai"):
        if isinstance(raw.get(key), bool):
            out[key] = raw[key]
    return out


def load_settings(text):
    try:
        return clean_settings(json.loads(text))
    except (TypeError, ValueError):
        return clean_settings({})


def prompt_id(prompt):
    """Kurze Kennung des Prompts fürs Log (zum Vergleichen verschiedener Prompts)."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]


def radius_for(context, cfg):
    """Umkreis (m) für einen Standort: nach seiner Umgebung (`context` aus site_context.py), sonst der
    feste Wert aus dem Fenster. Eine unbekannte/fehlende Umgebung zählt wie 'kein context'."""
    if cfg.get("auto_radius"):
        key = {"inside_op": "radius_inside", "near_op": "radius_near", "open_desert": "radius_desert"}.get(context)
        if key:
            return float(cfg[key])
    return float(cfg["radius_m"])


def zone_note(context, radius_m, tell_context=False, m_per_px=None):
    """Zeile für die KI: Umkreis, Maßstab und - nur wenn tell_context - die Umgebung. Standard ist AUS: die
    Umgebungsangabe ('inside a mining operation ...') drückte die Einschätzung in Betrieben Richtung
    'clean_mine' (5 von 6 echten Halden dort bekamen unter 20 %)."""
    parts = [f"Zone radius {radius_m:.0f} m"]
    if m_per_px:
        parts.append(f"image scale {m_per_px:.2f} m per pixel (a 3 m tire is about {3.0 / m_per_px:.0f} px across)")
    where = {"inside_op": "inside a mining operation", "near_op": "at the edge of a mining operation",
             "open_desert": "open desert away from mines"}.get(context, "") if tell_context else ""
    if where:
        parts.append(f"location: {where}")
    return "; ".join(parts)


def padded_box(xmin, ymin, xmax, ymax, radius=BUFFER_M, min_side=MIN_SIDE_M):
    """Umschließendes Rechteck + Umkreis; mindestens min_side Meter pro Seite (um die Mitte)."""
    xmin, ymin, xmax, ymax = xmin - radius, ymin - radius, xmax + radius, ymax + radius
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half_w, half_h = max(xmax - xmin, min_side) / 2, max(ymax - ymin, min_side) / 2
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def image_size(width_m, height_m, max_px=1024):
    """Pixelgröße mit gleichem Seitenverhältnis, längste Seite = max_px."""
    if width_m >= height_m:
        return max_px, max(1, round(max_px * height_m / width_m))
    return max(1, round(max_px * width_m / height_m)), max_px


def thinking_param(model, mode, effort=""):
    """Wert für `thinking` in der Anfrage. "default" = weglassen (die Claude-5-Modelle denken dann
    selbst, und das Thinking zählt zu max_tokens). "off": Sonnet 5.5 kennt nur "between_tools" (kein
    Thinking vorab), die übrigen "disabled"; Opus 5.5 lässt sich nicht abschalten, und bei effort
    xhigh/max lehnt die API das Abschalten ab - dann wird `thinking` weggelassen."""
    if mode != "off" or effort in ("xhigh", "max") or "opus-5-5" in model:
        return None
    return {"type": "between_tools"} if "sonnet-5-5" in model else {"type": "disabled"}


EXAMPLE_KINDS = {"dump": "CONFIRMED TIRE DUMP", "not_dump": "CONFIRMED NOT A DUMP"}


def example_blocks(examples, cache=False):
    """Inhaltsblöcke für die Referenzbilder, die VOR dem eigentlichen Bild stehen. Jedes Beispiel ist ein dict
    mit b64, kind ('dump' / 'not_dump'), caption. Reihenfolge und Text sind stabil (nach id sortiert), damit der
    Cache greift; der Cache-Haltepunkt sitzt am Ende der Beispiele (System-Prompt + Beispiele werden gelesen,
    nur das neue Bild kostet voll)."""
    if not examples:
        return []
    blocks = [{"type": "text", "text": (
        "Reference examples follow. Each was checked by a human reviewer. Use them to calibrate what tires and "
        "what look-alikes (dirt and rock heaps, shadows) look like at this scale. The new view may look different "
        "from all of them: judge what it shows, do not copy a label.")}]
    for n, ex in enumerate(examples, 1):
        label = EXAMPLE_KINDS.get(ex.get("kind"), "EXAMPLE")
        caption = str(ex.get("caption", "")).strip()
        blocks.append({"type": "text", "text": f"Reference example {n} - {label}" + (f": {caption}" if caption else "")})
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ex["b64"]}})
    end = {"type": "text", "text": "End of reference examples. Now classify the NEW view below."}
    if cache:
        end["cache_control"] = {"type": "ephemeral"}
    blocks.append(end)
    return blocks


def pick_examples(entries, max_n):
    """Wählt aus den aktiven Beispielen höchstens max_n aus, ausgewogen zwischen 'dump' und 'not_dump'
    (abwechselnd, neueste zuerst), füllt mit der anderen Art auf, wenn eine Art zu wenige hat. Ergebnis nach
    id sortiert (stabil für den Cache)."""
    if max_n <= 0:
        return []
    pools = {k: sorted([e for e in entries if e.get("kind") == k], key=lambda e: e["id"], reverse=True)
             for k in ("dump", "not_dump")}
    out, order = [], ["dump", "not_dump"]
    while len(out) < max_n and (pools["dump"] or pools["not_dump"]):
        for k in order:
            if pools[k] and len(out) < max_n:
                out.append(pools[k].pop(0))
    return sorted(out, key=lambda e: e["id"])


def build_payload(image_b64, media_type="image/png", context="", model=DEFAULT_MODEL, max_tokens=400,
                  system=None, temperature=None, effort="", thinking="default", cache=False, examples=None):
    """Anfrage-Körper für die Messages-API (reine Funktion, testbar)."""
    text = "Classify what is visible in this map view."
    if context.strip():
        text += f"\nContext: {context.strip()}"
    target = [{"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
              {"type": "text", "text": text}]
    content = example_blocks(examples, cache) + target if examples else target
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": ([{"type": "text", "text": system or SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
                   if cache else (system or SYSTEM_PROMPT)),
        "messages": [{"role": "user", "content": content}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if effort:
        payload["output_config"] = {"effort": effort}
    think = thinking_param(model, thinking, effort)
    if think is not None:
        payload["thinking"] = think
    return payload


_FILLER = re.compile(r"^\s*(?:the|this)\s+(?:zone|area|image|view|site|scene|polygon|map|region)"
                     r"(?:\s+\w+)?\s+(?:shows?|contains?|displays?|appears?\s+to\s+(?:show|contain)|"
                     r"looks?\s+like|is\s+showing|depicts?)\s*[:,]?\s*", re.I)


def tidy_reason(text):
    """Füllwörter am Anfang ('The zone shows ...') entfernen und den Satz groß beginnen lassen -
    auch wenn ein älterer gespeicherter Prompt sie noch erzeugt."""
    text = " ".join(str(text).split())
    cut = _FILLER.sub("", text)
    return (cut[:1].upper() + cut[1:]) if cut != text and cut else text


def parse_reply(response):
    """Messages-API-Antwort -> {'label','confidence','dump_location','reason'}.
    Toleriert Text vor/nach dem JSON (und Code-Zäune); bei Müll label='unsure'."""
    text = "".join(b.get("text", "") for b in response.get("content", []) if b.get("type") == "text")
    match = re.search(r"\{.*\}", text, re.S)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except ValueError:
            data = {}
    label = str(data.get("label", "unsure")).strip().lower()
    if label not in LABELS:
        label = "unsure"
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    try:
        dump_p = max(0.0, min(1.0, float(data["dump_probability"])))
    except (KeyError, TypeError, ValueError):      # älterer Prompt ohne das Feld
        dump_p = conf if label == "dump" else (1.0 - conf if label.startswith("clean") else 0.5)
    return {"ok": bool(data), "stop_reason": response.get("stop_reason", ""),
            "tokens_in": usage_of(response)[0], "tokens_out": usage_of(response)[1],
            "usage": usage_detail(response),
            "label": label, "confidence": conf, "dump_probability": dump_p,
            "dump_location": str(data.get("dump_location", "")),
            "reason": tidy_reason(data.get("reason", text.strip() if not data else ""))}


def _call_retrying(payload, api_key, timeout, opener, sleep, tries=4):
    """call_api mit Wiederholung bei Überlastung/Ratenlimit (429, 529, 503): 2, 4, 8 s Pause."""
    for attempt in range(tries):
        try:
            return call_api(payload, api_key, timeout, opener)
        except RuntimeError as exc:
            if attempt == tries - 1 or not re.match(r"API-Fehler (429|503|529)", str(exc)):
                raise
            sleep(2 ** (attempt + 1))


# USD je 1 Mio. Token (Eingabe, Ausgabe) laut Anthropic-Preisseite; Thinking zählt als Ausgabe.
# Reihenfolge: spezifischere Namen zuerst. Preise können sich ändern - in den Einstellungen überschreibbar.
PRICES = (("haiku-5-5", 0.10, 0.50), ("haiku-4-5", 1.0, 5.0), ("sonnet-5-5", 2.0, 10.0),
          ("sonnet-5", 2.0, 10.0), ("sonnet-4-6", 3.0, 15.0), ("opus-5-5", 4.0, 20.0),
          ("opus-5", 5.0, 25.0), ("opus-4", 5.0, 25.0), ("fable-5", 10.0, 50.0))


def price_per_mtok(model, override_in=0.0, override_out=0.0):
    """(Eingabe, Ausgabe) in USD je Mio. Token; override > 0 ersetzt den Tabellenwert; None = unbekannt."""
    if override_in > 0 and override_out > 0:
        return override_in, override_out
    for name, p_in, p_out in PRICES:
        if name in model:
            return p_in, p_out
    return None


def usage_detail(response):
    """Token einer Antwort: eingegeben (ohne Cache), Cache geschrieben, Cache gelesen, ausgegeben.
    Bei wiederholter Anfrage (run_check) die Summe beider Anfragen."""
    u = response.get("usage") or {}
    keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
    got = [int(u.get(k, 0) or 0) for k in keys]
    before = response.get("_usage_before") or (0, 0, 0, 0)
    return {n: g + b for n, g, b in zip(("input", "cache_write", "cache_read", "output"), got, before)}


def usage_of(response):
    """(Eingabe-, Ausgabe-Token) einer Antwort - Cache-Token zählen zur Eingabe."""
    d = usage_detail(response)
    return d["input"] + d["cache_write"] + d["cache_read"], d["output"]


def estimate_cost(model, tokens_in, tokens_out, override_in=0.0, override_out=0.0,
                  cache_write=0, cache_read=0):
    """Geschätzte Kosten in USD (None, wenn der Preis des Modells unbekannt ist). tokens_in = normal
    berechnete Eingabe-Token; Cache schreiben kostet 1,25x, Cache lesen 0,1x des Eingabepreises."""
    price = price_per_mtok(model, override_in, override_out)
    if price is None:
        return None
    return (tokens_in * price[0] + cache_write * price[0] * 1.25 + cache_read * price[0] * 0.1
            + tokens_out * price[1]) / 1e6


def cost_of(model, res, override_in=0.0, override_out=0.0):
    """Kosten eines geparsten Ergebnisses (benutzt seine Token-Aufschlüsselung)."""
    u = res.get("usage") or {"input": res.get("tokens_in", 0), "cache_write": 0, "cache_read": 0,
                             "output": res.get("tokens_out", 0)}
    return estimate_cost(model, u["input"], u["output"], override_in, override_out,
                         u["cache_write"], u["cache_read"])


def fmt_cost(usd):
    if usd is None:
        return "Preis unbekannt"
    return f"≈ ${usd:.4f}" if usd < 0.01 else f"≈ ${usd:.3f}"


def run_check(payload, api_key, timeout=90, opener=urllib.request.urlopen, sleep=time.sleep):
    """Anfrage senden; endet sie wegen max_tokens ohne lesbare Antwort (Thinking hat die Tokens
    verbraucht), einmal mit dem 4-fachen Limit (höchstens 32000) wiederholen."""
    resp = _call_retrying(payload, api_key, timeout, opener, sleep)
    if not parse_reply(resp)["ok"] and resp.get("stop_reason") == "max_tokens":
        bigger = dict(payload, max_tokens=min(32000, payload["max_tokens"] * 4))
        d = usage_detail(resp)
        first = (d["input"], d["cache_write"], d["cache_read"], d["output"])
        resp = _call_retrying(bigger, api_key, timeout, opener, sleep)
        resp["_retried_with"] = bigger["max_tokens"]
        resp["_usage_before"] = first                  # die erste, abgeschnittene Anfrage wurde auch berechnet
    return resp


def banner(result, thr_dump=0.5, thr_possible=0.2):
    """(Text, Farbe) des Ergebnis-Kastens. Die Prozentzahl ist die Wahrscheinlichkeit, dass eine Halde
    vorhanden ist: ab thr_dump 'Dump found' (rot), ab thr_possible 'Possible dump' (orange - Recall
    hat Vorrang, solche Standorte sollen nicht als 'kein Dump' durchrutschen), darunter 'No dump'."""
    if not result.get("ok", True):
        return "No answer (unreadable)", "#6d6d6d"
    p = result["dump_probability"]
    if p >= thr_dump:
        text, color = "Dump found", "#c62828"
    elif p >= thr_possible:
        text, color = "Possible dump", "#d9822b"
    else:
        text, color = "No dump", "#2e7d32"
    return f"{text} ({round(p * 100):02d}%)", color


def ai_note(result):
    """Notiztext: was die KI zu sehen glaubt, z.B. 'AI: clean_road, dump 0.04 - Haul road with ...'."""
    reason = " ".join(str(result.get("reason", "")).split())
    return f"AI: {result['label']}, dump {result['dump_probability']:.2f}" + (f" - {reason}" if reason else "")


def verdict_to_review(result, thr_dump=0.5, thr_possible=0.2):
    """KI-Ergebnis -> Wert der Spalte `review`: dump / partial / unsure / clean.
    Halde nur teilweise im Umriss = partial; Halde nur in der Umgebung (Beinahe-Treffer) = unsure.
    'clean' nur, wenn die KI sich wirklich sicher ist (unter thr_possible UND keine Halde gesehen):
    jeder Zweifel wird 'unsure', damit ein Umriss mit Halde nicht als Fehlalarm ins Training geht."""
    p, where = result["dump_probability"], result.get("dump_location", "")
    if p >= thr_dump:
        return {"partial": "partial", "nearby": "unsure"}.get(where, "dump")
    if p < thr_possible and where in ("", "none"):
        return "clean"
    return "unsure"


# Spalten, die das Plugin in der Standortschicht anlegt: die KI-Einschätzung bleibt in der Datei
# gespeichert (so kostet das erneute Öffnen eines Standorts keine Anfrage) und ai_decision hält fest, was
# daraus wurde: pending (nur eingeschätzt), accepted / denied (du), auto_clean (automatisch clean),
# spot_check (Auto-Clean-Kandidat, von dir zu prüfen), spot_ok / spot_miss (Ergebnis dieser Kontrolle),
# human_agree / human_differs (du hast selbst entschieden, mit/ohne Übereinstimmung).
AI_FIELDS = (("ai_label", "string"), ("ai_dump_p", "double"), ("ai_loc", "string"),
             ("ai_reason", "string"), ("ai_decision", "string"))


def call_api(payload, api_key, timeout=90, opener=urllib.request.urlopen):
    """POST an die Messages-API; wirft RuntimeError mit lesbarer Meldung."""
    req = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"content-type": "application/json", "x-api-key": api_key,
                 "anthropic-version": API_VERSION})
    try:
        with opener(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            detail = ""
        raise RuntimeError(f"API-Fehler {exc.code}: {detail or exc.reason}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Keine Verbindung zur API: {exc.reason}")


def log_row(path, row):
    """Eine Zeile an die CSV anhängen. Existiert die Datei schon, gilt ihre Kopfzeile (neue Spalten werden
    ignoriert, fehlende bleiben leer) - sonst würden Werte unter falschen Überschriften landen."""
    header = list(row)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            first = fh.readline().strip()
        if first:
            header = next(csv.reader([first]))
    new = not os.path.exists(path) or not os.path.getsize(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore", restval="")
        if new:
            writer.writeheader()
        writer.writerow(row)


def auto_clean_ok(res, context, cfg):
    """Darf dieser Standort ohne deine Prüfung als clean gelten? Alle Bedingungen müssen stimmen:
    Auto-Clean an, Umgebung erlaubt (Standard: nur open_desert), Antwort lesbar, Halden-Wahrscheinlichkeit
    höchstens auto_clean_p und keine Halde in der Zone gesehen."""
    allowed = [c for c in str(cfg.get("auto_clean_contexts", "")).split(",") if c]
    return bool(cfg.get("auto_clean") and res.get("ok") and context in allowed
                and res["dump_probability"] <= cfg["auto_clean_p"]
                and res.get("dump_location", "") in ("", "none")
                and not str(res.get("label", "")).startswith("dump"))


def spot_check_pick(site_id, every):
    """Jeder n-te Auto-Clean-Kandidat wird dir zur Kontrolle vorgelegt. Die Auswahl hängt nur von der
    site_id ab (nicht vom Zufall der Reihenfolge): wiederholbar, ohne Zählerstand speichern zu müssen."""
    if every <= 1:
        return True
    return int(hashlib.sha1(str(site_id).encode("utf-8")).hexdigest(), 16) % every == 0


def batch_outcome(res, context, site_id, cfg):
    """Was passiert mit dem Ergebnis eines Stapel-Standorts? auto_clean | spot_check | pending."""
    if not auto_clean_ok(res, context, cfg):
        return "pending"
    return "spot_check" if spot_check_pick(site_id, cfg["spot_check_every"]) else "auto_clean"


def human_followup(previous_decision, human_review, ai_verdict):
    """Neuer Wert für ai_decision, wenn du (Tasten 1-4) über einen Standort mit KI-Ergebnis entscheidest.
    Liefert None, wenn nichts zu vermerken ist. Daraus ergibt sich später die Fehlerquote des Auto-Clean:
    spot_ok / spot_miss bei Stichproben, sonst human_agree / human_differs."""
    if previous_decision == "spot_check":
        return "spot_ok" if human_review == "clean" else "spot_miss"
    if previous_decision == "pending":
        return "human_agree" if human_review == ai_verdict else "human_differs"
    return None


CONFLICT_LOW = 0.4        # dein Urteil dump/partial, die KI höchstens so sicher
CONFLICT_HIGH = 0.7       # dein Urteil clean, die KI mindestens so sicher


def find_conflicts(items, low=CONFLICT_LOW, high=CONFLICT_HIGH):
    """Widersprüche zwischen deinem Urteil und der KI: items = (fid, review, ai_dump_p oder None).
    Liefert [(fid, art, abstand)], der größte Widerspruch zuerst. art: 'dump_ki_niedrig' (du: dump/partial, KI <= low)
    oder 'clean_ki_hoch' (du: clean, KI >= high). Tippfehler beim Beurteilen fallen fast immer in diese Gruppen."""
    out = []
    for fid, review, p in items:
        if p is None:
            continue
        if review in ("dump", "partial") and p <= low:
            out.append((fid, "dump_ki_niedrig", 1.0 - p))
        elif review == "clean" and p >= high:
            out.append((fid, "clean_ki_hoch", p))
    out.sort(key=lambda t: -t[2])
    return out


def result_from_fields(label, dump_p, location, reason):
    """Gespeicherte Einschätzung (Spalten ai_*) wieder als Ergebnis - kostet keine Anfrage."""
    try:
        p = max(0.0, min(1.0, float(dump_p)))
    except (TypeError, ValueError):
        return None
    return {"ok": True, "stop_reason": "", "tokens_in": 0, "tokens_out": 0,
            "usage": {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0},
            "label": str(label or "unsure"), "confidence": 0.0, "dump_probability": p,
            "dump_location": str(location or ""), "reason": str(reason or ""), "cached": True}


def ai_overview(entries, thr_dump=0.5, thr_possible=0.2):
    """Überblick über KI-Ergebnisse. entries: (Halden-Wahrscheinlichkeit, ai_decision) je Standort.
    dump / possible / none nach denselben Grenzen wie der Kasten im Fenster; auto_clean und spot_check
    zählen zusätzlich (auto_clean sind Teil von none, spot_check ebenfalls)."""
    o = {"n": 0, "dump": 0, "possible": 0, "none": 0, "auto_clean": 0, "spot_check": 0}
    for p, decision in entries:
        o["n"] += 1
        o["dump" if p >= thr_dump else "possible" if p >= thr_possible else "none"] += 1
        if decision in ("auto_clean", "spot_check"):
            o[decision] += 1
    o["to_review"] = o["n"] - o["auto_clean"]
    return o


def fmt_overview(o):
    """Eine Zeile: wie viele Dumps vermutet, unsicher, negativ - und wie viele automatisch übersprungen."""
    return (f"{o['n']} geprüft:  Dump {o['dump']} · Möglich {o['possible']} · Kein Dump {o['none']}"
            f"  |  automatisch übersprungen {o['auto_clean']} · Stichprobe {o['spot_check']}"
            f"  |  für dich zu prüfen {o['to_review']}")


def audit_summary(decisions):
    """Zusammenfassung der ai_decision-Werte: wie oft wurde automatisch bereinigt und wie oft lag die KI
    in deinen Stichproben daneben? Text für das Fenster."""
    n = lambda k: sum(1 for d in decisions if d == k)       # noqa: E731
    ok, miss = n("spot_ok"), n("spot_miss")
    line = f"Auto-Clean {n('auto_clean')}, Stichprobe offen {n('spot_check')}"
    if ok + miss:
        line += f", Stichprobe geprüft {ok + miss}: {miss} falsch ({miss / (ok + miss):.0%})"
    return line


class ReviewQueue:
    """Reihenfolge-Logik ohne QGIS: ungeprüfte Standorte nach Rang, Überspringen
    hängt hinten an, Zurück stellt die letzte Entscheidung wieder vorne ein."""

    def __init__(self, items):
        """items: Liste von (fid, rank, review) oder (fid, rank, review, note);
        leeres review = noch ungeprüft."""
        ordered = sorted(items, key=lambda it: it[1])
        self.total = len(ordered)
        self.rank = {it[0]: it[1] for it in ordered}
        self.reviews = {it[0]: (it[2] or "") for it in ordered}
        self.notes = {it[0]: ((it[3] if len(it) > 3 else "") or "") for it in ordered}
        self.todo = deque(it[0] for it in ordered if not it[2])
        self.history = []                      # (fid, vorheriges Urteil, vorherige Notiz)

    @property
    def current(self):
        return self.todo[0] if self.todo else None

    def resort(self, ai_p=None):
        """Offene Standorte neu ordnen. ai_p: Funktion fid -> KI-Wahrscheinlichkeit (oder None); dann kommen Standorte mit
        Ergebnis zuerst (höchste Wahrscheinlichkeit vorne), danach die übrigen nach Rang. Ohne ai_p: wieder nach Rang."""
        def key(fid):
            p = ai_p(fid) if ai_p else None
            return (0, -p, self.rank[fid]) if p is not None else (1, 0.0, self.rank[fid])
        self.todo = deque(sorted(self.todo, key=key))

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

    def set_external(self, fid, value, note=None):
        """Urteil von außen setzen (Stapel-Auto-Clean): der Standort verlässt die Warteschlange, ohne dass
        die aktuelle Position wandert und ohne Eintrag in 'Zurück'."""
        if fid not in self.reviews:
            return False
        self.reviews[fid] = value
        if note is not None:
            self.notes[fid] = note
        if fid in self.todo:
            self.todo.remove(fid)
        return True

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

    def requeue(self, fids):
        """Schon beurteilte Standorte noch einmal vorne einreihen (in der gegebenen Reihenfolge). Das Urteil bleibt,
        bis du neu entscheidest; 'Überspringen' lässt es unverändert. Gibt die Anzahl zurück."""
        fids = [f for f in fids if f in self.reviews]
        for fid in reversed(fids):
            if fid in self.todo:
                self.todo.remove(fid)
            self.todo.appendleft(fid)
        return len(fids)

    def counts(self):
        out = {}
        for v in self.reviews.values():
            if v:
                out[v] = out.get(v, 0) + 1
        return out
