# OTR Review - QGIS plugin

One plugin instead of three console scripts (`qgis/load_candidate_sites.py`, `qgis/review_queue.py`,
`qgis/ai_classifier.py`, which stay as a fallback). Written for QGIS 3.28+ and QGIS 4 (Qt6).

## Install
1. `python qgis_plugin/build_zip.py` (or use the `otr_review.zip` you were sent).
2. QGIS: Plugins > Manage and Install Plugins > Install from ZIP > pick `otr_review.zip`.
3. Plugins > OTR Review > *Kandidatenliste laden ...* (pick `candidate_sites.gpkg`), then the review window opens.
   The toolbar button (tire icon) reopens it. The API key is read from `ANTHROPIC_API_KEY` or asked for once
   (same stored key as the old scripts); settings of the old scripts are taken over.

## One window
Review keys 1-7 as before. Below: AI check (with result box, accept / deny, token and cost line),
the Auto checkbox (current + next N sites are checked ahead), batch analysis and the settings button.

## Batch analysis and automatic clean
*Stapel starten* checks the next N (default 500) unreviewed sites that have no AI result yet, several requests in
parallel, with a cost limit (default $1; settings).
- Sites in `open_desert` (column `context`, see `src/site_context.py`) with a dump probability of at most 10 %
  and no dump seen are written as `review = clean`, `ai_decision = auto_clean`, note `... [auto]`.
- Every 10th such site (by site id) stays in your queue as a spot check (`ai_decision = spot_check`). Your decision
  on it is recorded as `spot_ok` / `spot_miss`; the window shows the miss rate.
- While the batch runs the window counts live: Dump / Möglich / Kein Dump, automatically skipped, spot checks, cost.
  When it ends a report shows the same plus how many are left for you. A standing overview line stays in the window.
- Everything else keeps its stored AI result (columns `ai_label`, `ai_dump_p`, `ai_loc`, `ai_reason`,
  `ai_decision = pending`), so reopening a site costs no request.
- *Test auf Auswahl* checks the sites you selected in QGIS (e.g. all with `review = dump / partial`) without
  changing anything and reports the lowest probabilities. Run it before trusting auto-clean.

`src/reviewed_to_labels.py` keeps `auto_clean` sites out of the hard negatives (they become ignore zones) unless you
pass `--trust-ai-clean`.

## Development
Logic without Qt lives in `otr_review/core.py` (tests: `tests/test_otr_core.py`). The Qt part is smoke-tested in a
headless QGIS (needs `python3-qgis`, runs against QGIS 3.34/Qt5):

    QT_QPA_PLATFORM=offscreen /usr/bin/python3.12 qgis_plugin/smoke_test.py
