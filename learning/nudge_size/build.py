"""Build the Nudge Size Lab page from the experiment logs.

    python learning/nudge_size/build.py <runs_dir> <out.html>
Run experiment.py first (see its docstring). Verdict text lives in verdicts.json and must match the real logs.
"""
import json, sys, glob, pathlib
here = pathlib.Path(__file__).parent
def rnd(o, d=4):
    if isinstance(o, float): return float(f"{o:.{d}g}")
    if isinstance(o, list): return [rnd(x, d) for x in o]
    if isinstance(o, dict): return {k: rnd(v, d) for k, v in o.items()}
    return o
runs = {json.load(open(f))["name"]: rnd(json.load(open(f))) for f in glob.glob(f"{sys.argv[1]}/*.json")}
t = (here / "template.html").read_text().replace("__RUNS__", json.dumps(runs)).replace("__VERDICTS__", (here / "verdicts.json").read_text())
pathlib.Path(sys.argv[2]).write_text(t); print("wrote", sys.argv[2], len(t))
