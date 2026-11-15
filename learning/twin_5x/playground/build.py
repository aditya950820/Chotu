"""Build the Talk to the Twins page: the template with twins.js inlined.

    python learning/twin_5x/playground/export.py --dir <ckpt folder> --out <folder>   # tokenizer + both twins
    python learning/twin_5x/playground/build.py <folder>/index.html
Publish index.html with tokenizer.json, manifest.json and twin140/twin700 .1.txt/.2.txt next to it
(export.py's .bin files as base64 text in two halves, since the page host serves no raw binary files).
"""
import pathlib, sys
here = pathlib.Path(__file__).parent
page = (here / "template.html").read_text().replace("__TWINS_JS__", (here / "twins.js").read_text())
pathlib.Path(sys.argv[1]).write_text(page)
print("wrote", sys.argv[1], len(page), "bytes")
