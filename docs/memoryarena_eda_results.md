# MemoryArena EDA Results

This note summarizes the Kaggle EDA run stored locally under:

```text
artifacts/memoryarena_eda/kaggle_run_2026-06-01/
```

The generated files are intentionally kept out of Git. The tracked notebook and helper code can reproduce them from the Hugging Face dataset.

## Dataset Scale

- Total tasks: 701
- Total subtasks: 4,850
- `bundled_shopping`: 150 tasks, 900 subtasks
- `progressive_search`: 221 tasks, 1,641 subtasks
- `group_travel_planner`: 270 tasks, 1,869 subtasks
- `formal_reasoning_math`: 40 tasks, 354 subtasks
- `formal_reasoning_phys`: 20 tasks, 86 subtasks

No task had an invalid `len(questions) != len(answers)` schema check.

## Proxy Difficulty

Average proxy difficulty score:

| subset | score |
| --- | ---: |
| `group_travel_planner` | 0.314 |
| `bundled_shopping` | 0.309 |
| `formal_reasoning_math` | 0.279 |
| `progressive_search` | 0.148 |
| `formal_reasoning_phys` | 0.081 |

This score is an EDA heuristic, not an official MemoryArena metric.

## Main Findings

- `group_travel_planner` has the strongest relational memory pressure: every task has dependency edges, with 7,550 extracted relation rows.
- `bundled_shopping` is highly structured: 900 answers, 694 unique ASINs, and 5.24 attributes per answer on average.
- `formal_reasoning_math` has the heaviest context length: mean total tokens per task is about 7,517 and mean background tokens per task is about 5,303.
- `progressive_search` has a strong final-question expansion pattern: final questions average about 122 tokens, while intermediate questions average about 29 tokens.
- `formal_reasoning_phys` is substantially smaller and lighter than `formal_reasoning_math` in both task length and token pressure.

## Recommended Memory Mechanisms

- `bundled_shopping`: structured key-value and attribute memory.
- `progressive_search`: constraint accumulation memory.
- `group_travel_planner`: entity-relation graph memory.
- `formal_reasoning_*`: symbolic, definition, and intermediate lemma/result memory.

## Notebook Follow-Ups

- Filter LaTeX boilerplate commands such as `\newcommand`, `\def`, `\begin`, and `\end` before notation-reuse analysis.
- Namespace travel graph nodes by `task_id` to avoid merging common person names across unrelated tasks.
- Canonicalize progressive-search answer entities instead of counting raw long answer strings.
- Group shopping categories semantically; the current category labels are close to one unique label per task.
