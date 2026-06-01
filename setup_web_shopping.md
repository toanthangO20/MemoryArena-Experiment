# WebShop Environment Setup

## Overview

The WebShop environment evaluates memory-augmented agents on multi-step product search and purchase tasks. An agent navigates a simulated online store using search and click actions to find products matching a set of attribute and price constraints. Purchases from earlier steps are stored in memory and retrieved to inform later ones.

- **Dataset**: Loaded automatically from HuggingFace ([ZexueHe/memoryarena](https://huggingface.co/datasets/ZexueHe/memoryarena), config `web_shopping`).
- **Actions**: `search[query]`, `click[element]` — exactly one per turn.
- **Supported LLM backends**: Any OpenAI-compatible endpoint.
- **Memory systems**: All systems supported by MemoryArena (`long_context`, `text-embedding-3-small`, `bm25`, `mirix`, `letta`, `mem0`, `memorag`, `graphrag`, `reasoningbank`, `zep`, etc.).

The shopping code path is self-contained under `run_shopping.py` and `env/env_systems/web_shopping_env/`. It does not import from `agentenv_webshop/`, `agentenv/`, `webshop/`, or `Webshop_Plus/`.

## 1. Install Dependencies

```bash
pip install -r env/env_systems/web_shopping_env/requirements-shopping.txt
python -m spacy download en_core_web_lg
```

A JDK is also required for the search engine. If `javac` is not already on `PATH`:

```bash
export JAVA_HOME=/path/to/jdk
export PATH="$JAVA_HOME/bin:$PATH"
```

## 2. Data Files

The runtime requires a local product database. Download the product database and place it at `data/shopping/`:

**Download link**: https://huggingface.co/datasets/ai-hyz/MemoryArena-product-db at HuggingFace.

After downloading, the product database should look like:

```
data/shopping/
├── items_shuffle.json          ← product list (ASIN, attributes, price)
├── items_ins_v2.json           ← generated item attributes/instructions
├── domain_data.json            ← domain metadata
├── search_engine/indexes-full/ ← pyserini search index
├── product_catalog/            ← per-category catalog JSONs
....
├── feat_conv.pt                ← **optional** image feature tensor
└── feat_ids.pt                 ← **optional** image URL/id mapping
```

`items_human_ins.json` is only needed when human-authored goals are enabled with `human_goals=True`. The default text-only launcher uses synthetic goals (`human_goals=0`) and does not load it. `feat_conv.pt` and `feat_ids.pt` are only needed when image observations are enabled with `get_image=1`. The default text-only launcher does not load them.

If your data lives in a non-default location, override the paths in the config file:

```json
"env_config": {
  "upstream_webshop_data_root": "/path/to/data/shopping",
  "product_catalog_dir":        "/path/to/data/shopping/product_catalog",
  "domain_data_path":           "/path/to/data/shopping/domain_data.json"
}
```

## 3. Configure

Each experiment is defined by a JSON config file under `configs/web_shopping_configs/`. Pre-built configs are provided for all memory systems.

Example (`configs/web_shopping_configs/long_context.json`):

```json
{
  "agent": {
    "model_name": "gpt-5-mini",
    "backend": "openai",
    "api_key": "<YOUR_API_KEY>",
    "base_url": "<YOUR_API_BASE_URL>"
  },
  "memory": {
    "use_step_memory": false,
    "memory_system_name": "long_context",
    "server_url": "http://0.0.0.0:8000"
  },
  "env": {
    "env_name": "webshop",
    "env_server_url": "http://0.0.0.0:8005",
    "env_config": {
      "bootstrap_upstream_env": true,
      "action_format": "react"
    }
  },
  "task_specific": {
    "task_category": "all",
    "task_file_limit": -1,
    "max_steps": 25,
    "split_steps": true,
    "resume": true
  },
  "output": {
    "output_dir": "results/shopping"
  }
}
```

**Fields you need to set**:

| Field | Description |
|-------|-------------|
| `agent.api_key` | Your LLM API key (or set the `OPENAI_API_KEY` env var) |
| `agent.base_url` | API endpoint URL (for OpenAI-compatible proxies) |
| `agent.model_name` | Model to use (e.g. `gpt-5-mini`, `gpt-4o`) |
| `memory.memory_system_name` | Memory system to use (`long_context` requires no extra service) |
| `memory.server_url` | URL where the memory server is running |
| `env.env_server_url` | URL where the environment server is running |
| `task_specific.task_category` | Product category to evaluate, or `"all"` |
| `task_specific.task_file_limit` | Max task files per category; `-1` = unlimited |
| `output.output_dir` | Root directory for results and interaction logs |

To run a different memory system or model, see `configs/web_shopping_configs/all_systems_reference.json`. It lists all supported agent × memory presets with the exact field values and required environment variables for each system. Copy the relevant `agent` and `memory` blocks into your config file to reproduce a specific combination.

**Available configs** (`configs/web_shopping_configs/`):

| Config | Memory System | Model |
|--------|--------------|-------|
| `long_context-gpt-5-mini.json` | long_context | gpt-5-mini |
| `long_context-gpt-4.1-mini.json` | long_context | gpt-4.1-mini |
| `long_context-claude-sonnet-4.json` | long_context | claude-sonnet-4-5-20250929 |
| `long_context-gemini-3-flash.json` | long_context | gemini-3-flash-preview |
| `bm25.json` | bm25 | gpt-5-mini |
| `text-embedding-3-small.json` | text-embedding-3-small | gpt-5-mini |
| `graphrag.json` | graphrag | gpt-5-mini |
| `memorag.json` | memorag | gpt-5-mini |
| `reasoningbank.json` | reasoningbank | gpt-5-mini |
| `mem0.json` | mem0 | gpt-5-mini |
| `mem0-g.json` | mem0-g | gpt-5-mini |
| `letta.json` | letta | gpt-5-mini |
| `mirix.json` | mirix | gpt-5-mini |

## 4. Start Servers

Open two separate terminals:

```bash
# Terminal 1: Environment server
python env/env_server.py

# Terminal 2: Memory server (skip if memory_system_name is "long_context")
python memory/server.py
```

By default `env_server.py` runs on port 8005 and `memory/server.py` on port 8000. Make sure these match the URLs in your config. The internal WebShop runtime (`:36004`) is bootstrapped automatically by `run_shopping.py` — you do not need to start it manually.

## 5. Run

You need to start the servers at first. 

```bash
python run_shopping.py --config configs/web_shopping_configs/long_context-gpt-5-mini.json
```

This will:

1. Load task files from HuggingFace (cached locally after the first download).
2. For each task, run the agent through all purchase steps sequentially, skipping tasks that already have output files when `resume: true`.
3. After all tasks complete, score each step with LLM-as-judge and write per-step rewards.
4. Print average reward and success rate per category.

Results are saved under `output_dir`, structured as `{category}/{run_tag}/`.

## 6. Compute Reward

`run_shopping.py` calls the reward script automatically at the end of a run. You can also run it standalone after the fact using `env/env_systems/web_shopping_env/compute_reward.py`.

**Score a single run directory:**

```bash
python env/env_systems/web_shopping_env/compute_reward.py \
    --run-dir results/shopping/beauty/gpt-5-mini-split-long_context
```

**Score all runs under a specific category:**

```bash
python env/env_systems/web_shopping_env/compute_reward.py \
    --category beauty
```

**Score every run under the results directory:**

```bash
python env/env_systems/web_shopping_env/compute_reward.py \
    --all
```

The script uses an LLM-as-judge (default `gpt-4o`) to evaluate attribute matches. To skip the LLM and use string-matching only:

```bash
python env/env_systems/web_shopping_env/compute_reward.py \
    --all --no-llm
```

Output is written to `reward_report.json` inside each run directory. Use `--force` to recompute even if a report already exists.
