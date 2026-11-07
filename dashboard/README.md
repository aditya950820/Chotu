# Dashboards

Source for the two shared Claude dashboards, so either of us can rebuild or change them.

## Twin day (`twin_day/`)

Live view of the Session 3 rehearsal: pipeline stages, live tiles (step, loss, exam bits per byte, speed, MFU, cost),
loss over steps, the fire-drill overlay, every run on one scale, the checklist (today's gates tick themselves from the logs),
alerts, samples, kill rules, GPU options and next steps.
Published as the Claude dashboard "Chotu Twin Day" (ask Adi to share it).

How the numbers get there:

1. On the GPU, `scripts/twin_day.sh` pushes every run's `metrics.jsonl`, `summary.json` and the step `status.jsonl`
   to the `live-runs` branch every 30 seconds (one commit, force-pushed), and the report plus logs to `twin-day` at the end.
2. `python dashboard/twin_day/sync.py twin.json` pulls `live-runs` and writes one long table (`k` = live, run, step,
   eval, stage, check, alert, sample).
3. Claude uploads `twin.json` to the dashboard; the page (`index.html`) splits it into views with `dash.calc`.

`plan.json` holds the GPU prices from our portal and Jarvislabs screenshots; full-run hours are estimates.

## Training cost by model size (`training_cost/`)

`python cost.py` estimates GPU-hours and cost for Chotu (181M), a 7B and a 100B model on every card on the portal:
compute ≈ 6 × parameters × tokens, each card's approximate peak bf16 speed, an assumed MFU, and the portal's on-demand price.
Estimates, not quotes. `index.html` is the page; the two CSVs are its data.
