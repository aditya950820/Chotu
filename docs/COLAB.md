# Using Google Colab

Colab runs **data prep (Session 2, `notebooks/session2_data.ipynb`)** and the **rehearsals (Sessions 3–4)**. The H100 run (Session 5) stays on a rented machine: Colab sessions time out and disconnect, which breaks a 15–25 hour unattended run.

## How it stays in sync

- **Code:** GitHub. Open `notebooks/colab_setup.ipynb` straight from GitHub each time and it pulls the latest `main`.
- **Files:** Google Drive at `/MyDrive/chotu`. The Colab machine's own disk is wiped when the session ends.
- **Keys:** Colab Secrets (🔑 in the left sidebar), stored in each person's own Google account, never in git.
- **Notes:** `docs/PROGRESS.md`, pushed from Colab (with a `GITHUB_TOKEN` secret) or from your laptop.

## Open the notebook

[Open in Colab](https://colab.research.google.com/github/aditya950820/Chotu/blob/main/notebooks/colab_setup.ipynb), or in Colab: File → Open notebook → GitHub → `aditya950820/Chotu`.
For a private repo, tick "Include private repos" and authorise GitHub once.

## One-time setup (each of you, in your own Google account)

1. Add Secrets, with *Notebook access* on: `HF_TOKEN`, `WANDB_API_KEY`, `GIT_NAME`, `GIT_EMAIL`, and optionally `GITHUB_TOKEN`.
2. Runtime → Change runtime type → **L4** or **A100**. This needs Colab Pro or compute units; the free T4 has no bf16.
3. Run all cells. Cell 6 must print four `[OK]` lines.

## Optional: let Claude drive Colab (Colab MCP)

Google's [colab-mcp](https://github.com/googlecolab/colab-mcp) lets a Claude app **on your Mac** edit and run cells in a Colab tab in your browser. It runs locally, so each of you installs it on your own machine:

```bash
pip install uv                       # once
claude mcp add colab-mcp -- uvx git+https://github.com/googlecolab/colab-mcp
```

Or, for the Claude desktop app: Settings → Developer → Edit Config, then add this under `mcpServers`:

```json
"colab-mcp": { "command": "uvx", "args": ["git+https://github.com/googlecolab/colab-mcp"], "timeout": 30000 }
```

Restart Claude and ask it to open a Colab connection. A Colab tab opens, signed in to the Google account in that browser; approve it within 60 seconds. Only one connection works at a time.
