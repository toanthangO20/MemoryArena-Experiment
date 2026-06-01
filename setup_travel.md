# Travel Planner Environment Setup

## Overview

The Travel Planner environment evaluates memory-augmented agents on multi-person group travel planning tasks. An agent plans trips for a sequence of people with chained constraints, using tool calls (flight search, restaurant search, accommodation search, etc.) via a ReAct loop. Previous travelers' plans are stored in memory and retrieved to inform future planning.

- **Dataset**: 270 groups, loaded automatically from HuggingFace ([ZexueHe/memoryarena](https://huggingface.co/datasets/ZexueHe/memoryarena), config `group_travel_planner`).
- **Tools**: FlightSearch, RestaurantSearch, AccommodationSearch, AttractionSearch, DistanceMatrix, CitySearch — all backed by local CSV databases.
- **Supported LLM backends**: OpenAI, Gemini, Anthropic.
- **Memory systems**: All systems supported by MemoryArena (`long_context`, `text-embedding-3-small`, `bm25`, `mirix`, `letta`, `mem0`, `memorag`, `graphrag`, `reasoningbank`, `zep`, etc.).

## 1. Install Dependencies

```bash
pip install -r env/env_systems/travel_planner_env/requirements.txt
```

## 2. Download Database

The tool APIs require a local database of flights, restaurants, accommodations, attractions, and distances. Most files are included in the repository, but `clean_Flights_2022.csv` is too large for Git and must be downloaded separately.

**Download link**: `https://drive.google.com/drive/folders/1fHBg6Ro4bSbTBgxeQZ6piYNtmqy8a_kK?usp=sharing`

After downloading, place the file at:

```
env/env_systems/travel_planner_env/database/flights/clean_Flights_2022.csv
```

The full database directory structure should look like:

```
env/env_systems/travel_planner_env/database/
├── flights/
│   └── clean_Flights_2022.csv          ← download this file
├── restaurants/
│   └── clean_restaurant_2022.csv
├── accommodations/
│   └── clean_accommodations_2022.csv
├── attractions/
│   └── attractions.csv
├── googleDistanceMatrix/
│   └── distance.csv
├── background/
│   ├── citySet_with_states.txt
│   └── citySet.txt
└── README.md
```

## 3. Configure

Each experiment is defined by a JSON config file under `configs/travel_planner_configs/`. Pre-built configs are provided for all memory systems.

Example (`configs/travel_planner_configs/text-embedding-3-small.json`):

```json
{
  "task_name": "travel_planner",
  "agent": {
    "model_name": "gpt-5-mini",
    "backend": "openai",
    "api_key": "<YOUR_API_KEY>",
    "base_url": "<YOUR_API_BASE_URL>"
  },
  "memory": {
    "use_step_memory": false,
    "memory_system_name": "text-embedding-3-small",
    "server_url": "http://0.0.0.0:8000"
  },
  "env": {
    "env_name": "travel_planner",
    "env_server_url": "http://0.0.0.0:8005",
    "env_config": {
      "judgement_mode": "none"
    }
  },
  "task_specific": {
    "max_steps": 30
  },
  "output": {
    "output_dir": "results/travel/text-embedding-3-small/gpt-5-mini",
    "log_dir": "results/travel/logs",
    "global_csv": "results/travel/results.csv"
  }
}
```

**Fields you need to set**:

| Field | Description |
|-------|-------------|
| `agent.api_key` | Your LLM API key |
| `agent.base_url` | API endpoint URL (for OpenAI-compatible proxies) |
| `agent.model_name` | Model to use (e.g. `gpt-5-mini`, `gemini-3-flash`, `claude-sonnet-4`) |
| `agent.backend` | `openai`, `gemini`, or `anthropic` |
| `memory.memory_system_name` | Memory system to use (or `none` for no memory) |
| `memory.server_url` | URL where the memory server is running |
| `env.env_server_url` | URL where the environment server is running |
| `output.output_dir` | Directory to save generated plans |

## 4. Start Servers

Open two separate terminals:

```bash
# Terminal 1: Environment server
python env/env_server.py

# Terminal 2: Memory server (required unless memory_system_name is "none" or "long_context")
python memory/server.py
```

Note: by default `env_server.py` runs on port 8005 and `memory/server.py` runs on port 8000. Make sure these match the URLs in your JSON config.

## 5. Run

```bash
python run_travel.py --config configs/travel_planner_configs/<your_config>.json
```

This will:

1. Load 270 travel planning groups from HuggingFace (cached locally after first download).
2. For each group, run the agent through all persons sequentially (skip groups that already have output files).
3. After all groups are done, combine individual outputs into a submission file.
4. Evaluate against ground truth and print PS (Person Success), SPS (Slot-level Person Success), and SR (Success Rate) metrics.

Results are appended to the `global_csv` file specified in the config.

## Available Configs

| Config | Memory System | Model |
|--------|--------------|-------|
| `none.json` | none | gpt-5-mini |
| `long_context-gpt-4.1-mini.json` | long_context | gpt-4.1-mini |
| `long_context-gpt-5-mini.json` | long_context | gpt-5-mini |
| `long_context-gemini-3-flash.json` | long_context | gemini-3-flash |
| `long_context-claude-sonnet-4.json` | long_context | claude-sonnet-4 |
| `text-embedding-3-small.json` | text-embedding-3-small | gpt-5-mini |
| `bm25.json` | bm25 | gpt-5-mini |
| `mirix.json` | mirix | gpt-5-mini |
| `letta.json` | letta | gpt-5-mini |
| `mem0.json` | mem0 | gpt-5-mini |
| `mem0-g.json` | mem0-g | gpt-5-mini |
| `memorag.json` | memorag | gpt-5-mini |
| `graphrag.json` | graphrag | gpt-5-mini |
| `reasoningbank.json` | reasoningbank | gpt-5-mini |
