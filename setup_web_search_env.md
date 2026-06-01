## BrowseComp-Plus Environment in `MemoryArena/env`

- **Dataset**: `Tevatron/browsecomp-plus`, `Tevatron/browsecomp-plus-corpus` on Hugging Face  

---

## 1. Environment Overview

The `env` directory provides a generic **Environment Hosting System** for `MemoryArena`, with a FastAPI server (`env_server.py`) and a Python client (`env_client.py`). BrowseComp-Plus is implemented as one of these environments.

### High-level architecture

- **Agent / Search script** (e.g., `run_search.py`) talks to:
  - `EnvironmentClient` → `env_server.py` (environment)
  - Optionally, a separate memory server (e.g., MemActBench on port 8000)
- **BrowseComp-Plus environment**:
  - Loads dataset files from `env/data/` (queries, ground truth, corpus).
  - Uses a searcher that reads FAISS indexes and ID maps from `env/embeddings/`.
  - Evaluates agent outputs against BrowseComp-Plus labels.

---

## 2. Downloading the BrowseComp-Plus Dataset

The upstream BrowseComp-Plus project provides encrypted data for queries, answers, and relevance judgments, plus a non‑obfuscated corpus.

### 2.1. Decrypting the benchmark data

From the **root** of the repository run:

```bash
pip install datasets  # ensure you have datasets installed

python env/env_systems/web_search_env/scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```

This produces:

- `data/browsecomp_plus_decrypted.jsonl` – decrypted queries, answers, and relevance judgments.
- `topics-qrels/queries.tsv` – TSV file of queries.

You may need to authenticate with Hugging Face:

```bash
huggingface-cli login
```

or pass an `hf_token` where supported.

### 2.2. Downloading the corpus

The actual corpus is **not obfuscated**. You can load it directly from Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
```

You can then export it into the environment’s data directory, for example:

```python
from datasets import load_dataset
import json

ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")

with open("MemoryArena/env/data/corpus.jsonl", "w") as f:
    for row in ds:
        f.write(json.dumps(row) + "\n")
```

In `MemoryArena`, the BrowseComp-Plus environment typically expects:

- `env/data/browsecomp_plus_decrypted.jsonl` (or equivalent ground‑truth file).
- `env/data/corpus.jsonl` – the exported corpus JSONL.

You can adjust paths in your configuration or scripts if you prefer a different layout.

---

## 3. Environment Dependencies

### 3.1. Environment server (`env/requirements.txt`)
(Better use Anaconda to do) 
Install these in a Python environment dedicated to the `env` server:

```bash
cd MemoryArena
pip install --upgrade pip
pip install -r env/requirements.txt
```

### 3.2. Additional packages

Depending on how you build and serve your indexes, you may also need (usually installed in the environment where you run search agents and `run_search.py`):

- `datasets` – To load BrowseComp-Plus datasets from Hugging Face.
- `faiss-cpu` – For dense retrieval.
- `openai` / other model SDKs – If your searcher uses external APIs to compute embeddings.
- `numpy`, `tqdm`, etc. – Utility libraries used by your indexing and search scripts.

These are usually installed in the same environment used by `run_search.py` and the searcher.

### 3.3. Using uv (recommended)

This project uses `uv` with Python 3.10 to manage the environment. Install uv by running:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more information, see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

Then set up the environment:

```bash
uv sync
source .venv/bin/activate
uv pip install --no-build-isolation flash-attn  # Needed for faiss
```

### 3.4. Java 21

This repo depends on Java 21. You can install it via conda:

```bash
conda install -c conda-forge openjdk=21
```

Or, if you have sudo access, install via apt:

```bash
sudo apt update
sudo apt install -y openjdk-21-jdk
```

---

## 4. Running the Memory Server

Install env/MemActBench
```pip install -r env/requirements.txt ```
start the environment server:

```bash
cd env/MemActBench
python server.py
```

By default, this starts a FastAPI app on:

- `http://0.0.0.0:8000`

Better initialize a new environment for the memory server, that is not the same one used in by `run_search.py` and the searcher,search_agent.

---

## 5. `env/embeddings/`: Local Indexes Used by the Environment

The **key difference** between this environment and the upstream BrowseComp-Plus repository is how indexes are handled.

### 5.1. Local index directory

The BrowseComp-Plus environment in `MemoryArena` uses:

- `MemoryArena/env/embeddings/`

as the place where **dense retrieval indexes** and associated metadata live. Typical contents:

- `shard0.index`, `shard1.index`, ... – FAISS index shards.
- `shard0_id_map.json`, `shard1_id_map.json`, ... – ID maps from FAISS IDs to corpus document IDs.
- Optional additional files (e.g., OpenAI embedding indexes, Tevatron `.pkl` files).

The searcher code (e.g., in `env/env_systems/searcher`) is configured to read from this directory. When `run_search.py` or an agent runs, it will:

1. Load the corpus from `env/data/corpus.jsonl`.
2. Load FAISS indexes from `env/embeddings/`.
3. Map retrieved document IDs back to corpus entries via the ID maps.

### 5.2. Important disclaimer about “pre‑built indexes”

In the **upstream BrowseComp-Plus README**, there is a section:

- “Downloading Pre-Built Indexes” using `bash scripts_build_index/download_indexes.sh`.


For this environment:

- The files in `env/embeddings/` are **your own locally built indexes** for BrowseComp-Plus.
- You may later decide to **upload** these indexes to another location (e.g., a Hugging Face repo) and write a small script to download them into `env/embeddings/`, but that is separate from the official pre-built indexes provided by the upstream project.

Whenever the upstream docs mention “pre-built indexes”, treat that as guidance or reference for index quality and evaluation, **not** as a guarantee that this environment is using those exact files.

---

## 6. Building Your Own Indexes for `env/embeddings/`

You are free to build indexes using the file env/env_systems/web_search_env/scripts_build_index/build_openai_embedding_index.py, which implements:

1. **Export corpus** to `env/data/corpus.jsonl` as shown earlier.
2. **Encode** documents into embeddings using your model of choice.
3. **Build FAISS indexes** over these embeddings (optionally sharded).
4. **Write ID maps** that map FAISS internal IDs back to the `docid` or equivalent field in `corpus.jsonl`.
5. **Save everything under** `env/embeddings/` with filenames that match what your searcher expects (e.g., `shard*.index` and `shard*_id_map.json`).

You can then configure the environment / searcher (or pass flags via `run_search.py`) to point at the correct glob or filenames, such as:
- `env/embeddings/shard*.index`

### 6.1. Downloading pre-built embeddings from Hugging Face

To get the FAISS index shards and ID maps without building them locally, use the download script (or the CLI one-liner below). This puts files into `env/env_systems/web_search_env/embeddings/`, which is where `run_search.py` and the web_search_env searcher expect them.

**Option A – Python script (recommended)**

```bash
# From MemoryArena repo root. Uses default repo joanna690/websearch-embeddings.
pip install huggingface_hub
python env/env_systems/web_search_env/download_embeddings_from_hf.py
```

**Option B – Hugging Face CLI**

```bash
mkdir -p env/env_systems/web_search_env/embeddings
huggingface-cli download joanna690/websearch-embeddings \
  --repo-type dataset \
  --local-dir env/env_systems/web_search_env/embeddings
```

After downloading, run `run_search.py` with the default `--index-path` and `--corpus-path` so the searcher finds these files.

## 7. Downloading `browsecomp_all_jsons.jsonl` from Hugging Face

`run_search.py` reads subqueries/correct answers from:

- `env/env_systems/web_search_env/data/browsecomp_all_jsons.jsonl`

Download this file from `joanna690/websearch-embeddings` and place it in the data directory:

```bash
mkdir -p env/env_systems/web_search_env/data
huggingface-cli download joanna690/websearch-embeddings \
  --repo-type dataset \
  --include "browsecomp_all_jsons.jsonl" \
  --local-dir env/env_systems/web_search_env/data
```

You can quickly verify:

```bash
ls env/env_systems/web_search_env/data/browsecomp_all_jsons.jsonl
```

---

## 8. Relationship to `run_search.py` and Agents

The `env` directory is focused on **hosting environments**; the actual search experiments (progressive search, final‑only search, memory integration, etc.) are typically orchestrated by:

- `MemoryArena/run_search.py`
- Agent code in `MemoryArena/agent/`

In a typical setup:

1. You start the environment server:

   ```bash
   cd MemoryArena/env
   python env_server.py
   ```
By default, this starts a FastAPI app on:

- `http://0.0.0.0:8001`

2. You ensure:
   - Data is in `env/data/` (decrypted BrowseComp-Plus files, `corpus.jsonl`).
   - Indexes are in `env/embeddings/`.


The agent side is documented separately (e.g., in the top‑level `MemoryArena/README` or the search‑only README you use for queries), while this file focuses on how **the environment itself** is configured and how it uses `env/embeddings/`.

---

## 9. Summary

- The `MemoryArena/env` directory hosts a **BrowseComp-Plus environment** alongside other environments.
- Dataset download and decryption follow the **upstream BrowseComp-Plus instructions**, but you typically place resulting data under `env/data/`.
- The environment uses **local FAISS indexes** stored in `env/embeddings/`; these are **not** the official pre-built indexes from the upstream project, but your own builds.
- Agents and scripts talk to the environment via HTTP/JSON using `env/env_client.py` and `env/env_server.py`.

