"""Build the Ask the Crowd page (Lesson 4) from the experiment logs.

    python learning/batch_size/compass.py <data> runs/compass_start.json
    python learning/batch_size/compass.py <data> runs/compass_trained.json --ckpt <tiny_after_256.pt>
    (race runs: learning/nudge_size/experiment.py with --batch 2/8/32/128, same 2,048 windows, --eval-every-windows 128)
    python learning/batch_size/build.py learning/batch_size/runs <out.html>
"""
import json, sys, pathlib
here = pathlib.Path(__file__).parent; R = pathlib.Path(sys.argv[1])
def rnd(o):
    if isinstance(o, float): return float(f"{o:.4g}")
    if isinstance(o, list): return [rnd(x) for x in o]
    if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
    return o
comp = {k: rnd(json.load(open(R / f"compass_{k}.json"))) for k in ("start", "trained")}
race = [rnd({k: v for k, v in json.load(open(R / f"race_b{b}.json")).items() if k in ("batch", "steps", "val", "name")}) for b in (2, 8, 32, 128)]
t = (here / "template.html").read_text().replace("__COMPASS__", json.dumps(comp)).replace("__RACE__", json.dumps(race)).replace("__TEXT__", (here / "text.json").read_text())
pathlib.Path(sys.argv[2]).write_text(t); print("wrote", sys.argv[2], len(t))
