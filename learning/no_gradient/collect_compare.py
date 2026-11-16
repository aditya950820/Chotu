"""Ask the running server.py for both models' answers to 40 prompts x 4 settings (for the static Counting vs Learning page).
    python learning/no_gradient/collect_compare.py   (server.py must be running on port 8800)
Then build the page: compare_template.html with __DATA__ replaced by the JSON."""
import json, urllib.request, time
PROMPTS = {
 "Facts": ["The capital of France is", "Water boils at", "In 1969, the first", "The largest planet in our solar system is", "The human heart has", "Mount Everest is", "The speed of light is", "Shakespeare wrote"],
 "Science": ["Photosynthesis is the process by which", "DNA is", "Gravity is the force that", "An atom is made of", "Climate change is caused by", "Vaccines work by", "The brain is", "Electricity flows"],
 "How to": ["To make a cup of tea, you", "The best way to learn a new language is", "To stay healthy, you should", "If you want to save money,", "To write a good essay,", "The first step in solving a problem is", "To grow tomatoes,", "When you feel stressed,"],
 "History & society": ["The Roman Empire", "World War II ended in", "The Industrial Revolution", "Democracy is", "The printing press", "Mahatma Gandhi", "The United Nations was founded", "Ancient Egypt"],
 "Stories & open": ["Once upon a time, there was a", "The old man looked out of the window and", "My favourite thing about school is", "Dear friend, I am writing to tell you", "The most important thing about exercise is", "Artificial intelligence will", "The future of education", "In my opinion, the best book"],
}
def post(path, body):
    r = urllib.request.Request("http://localhost:8800" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())
out = []
t0 = time.time()
for cat, ps in PROMPTS.items():
    for p in ps:
        item = dict(category=cat, prompt=p, runs={})
        item["next"] = post("/api/next", dict(prompt=p))
        for key, temp, seed in [("top", 0.0, 1), ("s1", 0.8, 1), ("s2", 0.8, 2), ("s3", 0.8, 3)]:
            r = post("/api/write", dict(prompt=p, temp=temp, rep=1.2, length=50, seed=seed))
            item["runs"][key] = {"nograd": [[x["t"], x["deep"], round(x["p"], 4)] for x in r["nograd"]], "twin": [[x["t"], round(x["p"], 4)] for x in r["twin"]]}
        out.append(item)
        print(f"{len(out):2d} {time.time()-t0:5.0f}s {p}", flush=True)
json.dump(out, open("learning/no_gradient/runs/writing/compare_40prompts.json", "w"))
