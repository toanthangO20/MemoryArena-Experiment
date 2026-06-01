# Kaggle MemoryArena EDA Notebook

Notebook: `notebooks/kaggle_memoryarena_eda.ipynb`

## Run on Kaggle

1. Create a new Kaggle Notebook.
2. Turn on **Internet** in notebook settings.
3. Upload or paste `notebooks/kaggle_memoryarena_eda.ipynb`.
4. Run all cells.

The notebook does not require any Kaggle Input dataset. It clones this repository into `/kaggle/working/MemoryArena-Experiment` and downloads the MemoryArena dataset directly from Hugging Face with:

```python
datasets.load_dataset("ZexueHe/memoryarena", config_name)
```

Loaded configs:

- `bundled_shopping`
- `progressive_search`
- `group_travel_planner`
- `formal_reasoning_math`
- `formal_reasoning_phys`

## Outputs

All generated artifacts are written to:

```text
/kaggle/working/memoryarena_eda_outputs
```

Main outputs:

- `tables/task_level_summary.csv`
- `tables/subtask_level_summary.csv`
- `tables/shopping_answers.csv`
- `tables/travel_edges.csv`
- `tables/memory_units.csv`
- `tables/subset_summary.csv`
- `tables/difficulty_summary.csv`
- `figures/*.png`
- `figures/*.html`
- `EDA_REPORT.md`
- `/kaggle/working/memoryarena_eda_outputs.zip`

The proxy difficulty score in the notebook is a transparent heuristic for EDA, not an official MemoryArena metric.

