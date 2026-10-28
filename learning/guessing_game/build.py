"""Rebuild the Guessing Game page from real data.

    python learning/guessing_game/landmarks.py <workdir> 8   # needs the 14 val shards in <workdir>/data/shards
    python learning/guessing_game/rounds.py <workdir>
    python learning/guessing_game/build.py <workdir>        # writes <workdir>/guessing_game.html
"""
import json, sys, pathlib
W = pathlib.Path(sys.argv[1]); here = pathlib.Path(__file__).parent
def rnd(o):
    if isinstance(o, float): return float(f"{o:.4g}")
    if isinstance(o, list): return [rnd(x) for x in o]
    if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
    return o
t = (here / "template.html").read_text()
t = t.replace("__DATA__", json.dumps(rnd(json.load(open(W / "rounds.json"))), ensure_ascii=False).replace("</", "<\\/"))
t = t.replace("__LAND__", json.dumps(json.load(open(W / "landmarks.json"))))
(W / "guessing_game.html").write_text(t); print("wrote", W / "guessing_game.html")
