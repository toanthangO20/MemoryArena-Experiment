"""Utilities for Kaggle-friendly MemoryArena EDA.

The helpers in this module intentionally avoid project-local data files.
They load the public Hugging Face dataset, normalize the five benchmark
subsets into task-level and subtask-level tables, and provide lightweight
heuristics for memory-pressure analysis.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASET_NAME = "ZexueHe/memoryarena"
DEFAULT_CONFIGS = [
    "bundled_shopping",
    "progressive_search",
    "group_travel_planner",
    "formal_reasoning_math",
    "formal_reasoning_phys",
]

CONSTRAINT_KEYWORDS = [
    "which",
    "where",
    "when",
    "born",
    "graduated",
    "located",
    "stated",
    "interview",
    "between",
    "after",
    "before",
    "as of",
    "given all",
    "above constraints",
    "what is the full name",
    "compatible",
    "budget",
    "highest",
    "lowest",
    "highest-rated",
    "lowest-priced",
    "highest-priced",
    "previous",
    "ground truth",
]

TRAVEL_KEYWORDS = [
    "join",
    "joining",
    "same",
    "together",
    "cheaper",
    "more expensive",
    "rating",
    "cuisine",
    "room",
    "accommodation",
    "breakfast",
    "lunch",
    "dinner",
    "transportation",
    "attraction",
]

SLOT_KEYWORDS = [
    "transportation",
    "breakfast",
    "lunch",
    "dinner",
    "attraction",
    "accommodation",
]

FORMAL_SUBSETS = {"formal_reasoning_math", "formal_reasoning_phys"}
_ENCODER: Any = None


def safe_len(value: Any) -> int:
    """Return length for common containers, treating missing values as zero."""
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    try:
        return len(value)
    except TypeError:
        return 1


def to_text(value: Any, max_chars: int | None = None) -> str:
    """Convert arbitrary nested values to stable text."""
    if value is None:
        text = ""
    elif isinstance(value, float) and math.isnan(value):
        text = ""
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    else:
        text = str(value)
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def as_list(value: Any) -> list[Any]:
    """Normalize strings, dictionaries, tuples, arrays, and missing values."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    return [value]


def type_name(value: Any) -> str:
    if value is None:
        return "None"
    return type(value).__name__


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _get_encoder() -> Any:
    global _ENCODER
    if _ENCODER == "fallback":
        return None
    if _ENCODER is not None:
        return _ENCODER
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
        return _ENCODER
    except Exception:
        _ENCODER = "fallback"
        return None


def estimate_tokens(value: Any) -> int:
    """Estimate token length with tiktoken when available, else regex tokens."""
    text = to_text(value)
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def count_keyword_constraints(text: Any, keywords: Iterable[str] | None = None) -> int:
    """Count approximate constraint keywords and phrases in a text."""
    if keywords is None:
        keywords = CONSTRAINT_KEYWORDS
    lowered = to_text(text).lower()
    total = 0
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        total += len(re.findall(pattern, lowered))
    return total


def load_memoryarena_subset(
    config_name: str,
    dataset_name: str = DATASET_NAME,
    split: str = "test",
) -> list[dict[str, Any]]:
    """Load one MemoryArena subset from Hugging Face as a list of dicts."""
    from datasets import DatasetDict, load_dataset

    dataset = load_dataset(dataset_name, config_name)
    if isinstance(dataset, DatasetDict):
        if split in dataset:
            table = dataset[split]
        else:
            first_split = next(iter(dataset.keys()))
            table = dataset[first_split]
    else:
        table = dataset
    return [dict(row) for row in table]


def load_all_memoryarena_subsets(
    configs: Iterable[str] = DEFAULT_CONFIGS,
    dataset_name: str = DATASET_NAME,
    split: str = "test",
) -> dict[str, list[dict[str, Any]]]:
    """Load all requested MemoryArena configs."""
    return {
        config: load_memoryarena_subset(config, dataset_name=dataset_name, split=split)
        for config in configs
    }


def _row_id(record: dict[str, Any], fallback: int) -> Any:
    return record.get("id", fallback)


def _aligned_value(values: list[Any], idx: int, default: Any = "") -> Any:
    if not values:
        return default
    if idx < len(values):
        return values[idx]
    if len(values) == 1:
        return values[0]
    return default


def _all_text(values: Iterable[Any]) -> str:
    return "\n".join(to_text(value) for value in values if to_text(value))


def answer_structure_type(value: Any) -> str:
    """Classify answer shape with lightweight heuristics."""
    if value is None:
        return "missing"
    if isinstance(value, float) and math.isnan(value):
        return "missing"
    if isinstance(value, dict):
        if "target_asin" in value:
            return "shopping_answer_dict"
        if {"days", "transportation", "accommodation"} & set(value):
            return "travel_day_dict"
        return "dict"
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        items = as_list(value)
        if not items:
            return "empty_list"
        item_types = {answer_structure_type(item) for item in items}
        if item_types <= {"shopping_answer_dict"}:
            return "shopping_answer_list"
        if item_types <= {"travel_day_dict", "dict"} and any(
            isinstance(item, dict) and "days" in item for item in items
        ):
            return "travel_plan_list"
        if len(item_types) == 1:
            return f"list_of_{next(iter(item_types))}"
        return "mixed_list"
    text = to_text(value).strip()
    if not text:
        return "empty_text"
    token_count = estimate_tokens(text)
    latex_hits = len(extract_latex_terms(text))
    lowered = text.lower()
    if latex_hits >= 3 or "\\frac" in text or "$" in text:
        return "formula_heavy"
    if any(marker in lowered for marker in ["proof", "therefore", "hence", "lemma"]):
        return "proof_or_explanation"
    if token_count <= 20:
        return "short_text"
    if token_count >= 150:
        return "long_explanation"
    return "text"


def answer_structure_complexity(value: Any) -> int:
    mapping = {
        "missing": 0,
        "empty_text": 0,
        "empty_list": 0,
        "short_text": 1,
        "text": 2,
        "dict": 2,
        "shopping_answer_dict": 3,
        "shopping_answer_list": 4,
        "travel_day_dict": 3,
        "travel_plan_list": 5,
        "list_of_short_text": 3,
        "list_of_text": 4,
        "mixed_list": 5,
        "formula_heavy": 5,
        "long_explanation": 5,
        "proof_or_explanation": 6,
    }
    return mapping.get(answer_structure_type(value), 3)


def extract_latex_terms(text: Any, top_k: int | None = None) -> list[str]:
    """Extract LaTeX-like commands, symbolic terms, and math-heavy tokens."""
    text_value = to_text(text)
    if not text_value:
        return []
    commands = re.findall(r"\\[A-Za-z]+", text_value)
    symbolish = re.findall(
        r"[A-Za-z][A-Za-z0-9]*(?:[_^]\{?[A-Za-z0-9\\]+\}?)+",
        text_value,
    )
    common_math = re.findall(r"\b(?:frac|sum|int|operatorname|Spec|Hom|ker|dim)\b", text_value)
    terms = commands + symbolish + common_math
    if top_k is None:
        return terms
    return [term for term, _ in Counter(terms).most_common(top_k)]


def _extract_shopping_attrs(record: dict[str, Any]) -> list[str]:
    attrs: list[str] = []
    for answer in as_list(record.get("answers")):
        if isinstance(answer, dict):
            attrs.extend(to_text(attr).strip() for attr in as_list(answer.get("attributes")))
    return [attr for attr in attrs if attr]


def _extract_shopping_asins(record: dict[str, Any]) -> list[str]:
    asins: list[str] = []
    for answer in as_list(record.get("answers")):
        if isinstance(answer, dict) and answer.get("target_asin"):
            asins.append(to_text(answer.get("target_asin")))
    return asins


def _extract_person_names_from_questions(questions: Iterable[Any]) -> list[str]:
    names: list[str] = []
    for question in questions:
        match = re.search(r"\bI am ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", to_text(question))
        if match:
            names.append(match.group(1).strip())
    return names


def _base_person_name(record: dict[str, Any]) -> str | None:
    base_person = record.get("base_person")
    if isinstance(base_person, dict):
        name = base_person.get("name")
        if name:
            return to_text(name)
        query = base_person.get("query")
        match = re.search(r"\bI am ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", to_text(query))
        if match:
            return match.group(1).strip()
    return None


def _entity_or_attribute_count(subset: str, record: dict[str, Any]) -> int:
    if subset == "bundled_shopping":
        return len(set(_extract_shopping_asins(record))) + len(set(_extract_shopping_attrs(record)))
    if subset == "group_travel_planner":
        names = [_base_person_name(record)] + _extract_person_names_from_questions(
            as_list(record.get("questions"))
        )
        return len({name for name in names if name})
    if subset in FORMAL_SUBSETS:
        terms = []
        for value in as_list(record.get("questions")) + as_list(record.get("answers")) + as_list(
            record.get("backgrounds")
        ):
            terms.extend(extract_latex_terms(value))
        paper_name = record.get("paper_name")
        return len(set(terms)) + (1 if paper_name else 0)
    answers = {to_text(answer, max_chars=200) for answer in as_list(record.get("answers"))}
    return len({answer for answer in answers if answer})


def _dependency_reference_count(record: dict[str, Any]) -> int:
    questions = as_list(record.get("questions"))
    known_names = []
    base_name = _base_person_name(record)
    if base_name:
        known_names.append(base_name)
    count = 0
    for question in questions:
        text = to_text(question)
        count += sum(1 for name in known_names if re.search(r"\b" + re.escape(name) + r"\b", text))
        current = _extract_person_names_from_questions([question])
        known_names.extend(name for name in current if name not in known_names)
    return count


def build_task_df(raw_records: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Build one row per task/record across all MemoryArena subsets."""
    rows: list[dict[str, Any]] = []
    for subset, records in raw_records.items():
        for fallback_id, record in enumerate(records):
            questions = as_list(record.get("questions"))
            answers = as_list(record.get("answers"))
            backgrounds = as_list(record.get("backgrounds"))
            question_text = _all_text(questions)
            answer_text = _all_text(answers)
            background_text = _all_text(backgrounds)
            row = {
                "subset": subset,
                "id": _row_id(record, fallback_id),
                "num_questions": len(questions),
                "num_answers": len(answers),
                "num_backgrounds": len(backgrounds),
                "has_backgrounds": bool(background_text.strip()),
                "has_base_person": record.get("base_person") is not None,
                "has_paper_name": record.get("paper_name") is not None,
                "category": record.get("category"),
                "paper_name": record.get("paper_name"),
                "base_person_name": _base_person_name(record),
                "questions_type": type_name(record.get("questions")),
                "answers_type": type_name(record.get("answers")),
                "backgrounds_type": type_name(record.get("backgrounds")),
                "schema_valid": len(questions) == len(answers),
                "total_question_chars": len(question_text),
                "total_answer_chars": len(answer_text),
                "total_background_chars": len(background_text),
                "total_question_tokens": estimate_tokens(question_text),
                "total_answer_tokens": estimate_tokens(answer_text),
                "total_background_tokens": estimate_tokens(background_text),
                "answer_structure_type": answer_structure_type(record.get("answers")),
                "answer_structure_complexity": answer_structure_complexity(record.get("answers")),
                "constraint_keyword_count": count_keyword_constraints(question_text),
                "entity_or_attribute_count": _entity_or_attribute_count(subset, record),
                "dependency_reference_count": _dependency_reference_count(record),
            }
            row["total_tokens_est"] = (
                row["total_question_tokens"]
                + row["total_answer_tokens"]
                + row["total_background_tokens"]
            )
            row["proxy_difficulty_score"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def build_subtask_df(raw_records: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Build one row per session/subtask across all MemoryArena subsets."""
    rows: list[dict[str, Any]] = []
    for subset, records in raw_records.items():
        for fallback_id, record in enumerate(records):
            task_id = _row_id(record, fallback_id)
            questions = as_list(record.get("questions"))
            answers = as_list(record.get("answers"))
            backgrounds = as_list(record.get("backgrounds"))
            n = max(len(questions), len(answers), len(backgrounds))
            cumulative_question_tokens = 0
            cumulative_context_tokens = 0
            for idx in range(n):
                question = _aligned_value(questions, idx)
                answer = _aligned_value(answers, idx)
                background = _aligned_value(backgrounds, idx)
                question_tokens = estimate_tokens(question)
                answer_tokens = estimate_tokens(answer)
                background_tokens = estimate_tokens(background)
                cumulative_question_tokens += question_tokens
                cumulative_context_tokens += question_tokens + answer_tokens + background_tokens
                rows.append(
                    {
                        "subset": subset,
                        "task_id": task_id,
                        "subtask_idx": idx,
                        "question": to_text(question),
                        "answer": to_text(answer),
                        "background": to_text(background),
                        "question_chars": len(to_text(question)),
                        "answer_chars": len(to_text(answer)),
                        "background_chars": len(to_text(background)),
                        "question_tokens": question_tokens,
                        "answer_tokens": answer_tokens,
                        "background_tokens": background_tokens,
                        "cumulative_question_tokens": cumulative_question_tokens,
                        "cumulative_context_tokens": cumulative_context_tokens,
                        "question_keyword_constraint_count": count_keyword_constraints(question),
                        "answer_structure_type": answer_structure_type(answer),
                    }
                )
    return pd.DataFrame(rows)


def extract_shopping_answers(raw_records: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in raw_records.get("bundled_shopping", []):
        task_id = record.get("id")
        category = record.get("category")
        for idx, answer in enumerate(as_list(record.get("answers"))):
            if not isinstance(answer, dict):
                continue
            attrs = [to_text(attr).strip() for attr in as_list(answer.get("attributes")) if to_text(attr)]
            rows.append(
                {
                    "task_id": task_id,
                    "subtask_idx": idx,
                    "target_asin": answer.get("target_asin"),
                    "attributes": attrs,
                    "attributes_text": "|".join(attrs),
                    "num_attributes": len(attrs),
                    "category": category,
                }
            )
    return pd.DataFrame(rows)


def compute_attribute_cooccurrence(
    shopping_answers_df: pd.DataFrame,
    top_k: int = 30,
) -> pd.DataFrame:
    if shopping_answers_df.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])
    attr_counts = Counter()
    for attrs in shopping_answers_df["attributes"]:
        attr_counts.update(str(attr).lower() for attr in as_list(attrs) if str(attr).strip())
    top_attrs = {attr for attr, _ in attr_counts.most_common(top_k)}
    edge_counts: Counter[tuple[str, str]] = Counter()
    for attrs in shopping_answers_df["attributes"]:
        normalized = sorted({str(attr).lower() for attr in as_list(attrs) if str(attr).lower() in top_attrs})
        for source, target in combinations(normalized, 2):
            edge_counts[(source, target)] += 1
    return pd.DataFrame(
        [{"source": source, "target": target, "weight": weight} for (source, target), weight in edge_counts.items()]
    )


def build_progressive_search_df(raw_records: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    patterns = [
        "given all",
        "above constraints",
        "what is the full name",
        "what is the",
        "as of",
    ]
    rows: list[dict[str, Any]] = []
    for record in raw_records.get("progressive_search", []):
        questions = as_list(record.get("questions"))
        answers = as_list(record.get("answers"))
        answer_texts = [to_text(answer).strip() for answer in answers]
        intermediate_answers = answer_texts[:-1] if len(answer_texts) > 1 else answer_texts
        final_question = to_text(questions[-1]) if questions else ""
        rows.append(
            {
                "task_id": record.get("id"),
                "num_subqueries": len(questions),
                "unique_answer_count": len({answer for answer in answer_texts if answer}),
                "all_intermediate_answers_same": len({answer for answer in intermediate_answers if answer}) <= 1,
                "final_question_patterns": "|".join(
                    pattern for pattern in patterns if pattern in final_question.lower()
                ),
                "final_question_tokens": estimate_tokens(final_question),
                "mean_intermediate_question_tokens": np.mean(
                    [estimate_tokens(question) for question in questions[:-1]]
                )
                if len(questions) > 1
                else 0.0,
                "constraint_keyword_count": count_keyword_constraints(_all_text(questions)),
            }
        )
    return pd.DataFrame(rows)


def build_travel_df(raw_records: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in raw_records.get("group_travel_planner", []):
        questions = as_list(record.get("questions"))
        base_person = record.get("base_person") if isinstance(record.get("base_person"), dict) else {}
        base_name = _base_person_name(record)
        daily_plans = as_list(base_person.get("daily_plans")) if isinstance(base_person, dict) else []
        question_names = _extract_person_names_from_questions(questions)
        all_question_text = _all_text(questions)
        row = {
            "task_id": record.get("id"),
            "base_person_name": base_name,
            "num_travelers": 1 + len(questions),
            "num_trip_days": len(daily_plans),
            "question_person_names": "|".join(question_names),
            "constraint_keyword_count": count_keyword_constraints(all_question_text, TRAVEL_KEYWORDS),
        }
        for keyword in TRAVEL_KEYWORDS:
            row[f"kw_{keyword.replace(' ', '_')}"] = count_keyword_constraints(all_question_text, [keyword])
        for slot in SLOT_KEYWORDS:
            row[f"slot_{slot}"] = count_keyword_constraints(all_question_text, [slot])
        rows.append(row)
    return pd.DataFrame(rows)


def extract_travel_edges(raw_records: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in raw_records.get("group_travel_planner", []):
        task_id = record.get("id")
        known_names: list[str] = []
        base_name = _base_person_name(record)
        if base_name:
            known_names.append(base_name)
        for idx, question in enumerate(as_list(record.get("questions"))):
            text = to_text(question)
            current_names = _extract_person_names_from_questions([text])
            current = current_names[0] if current_names else f"traveler_{idx + 1}"
            for prior in known_names:
                if re.search(r"\b" + re.escape(prior) + r"\b", text):
                    rows.append(
                        {
                            "task_id": task_id,
                            "subtask_idx": idx,
                            "source_person": current,
                            "target_person": prior,
                            "question_excerpt": to_text(text, max_chars=240),
                        }
                    )
            for name in current_names:
                if name not in known_names:
                    known_names.append(name)
    return pd.DataFrame(rows)


def build_formal_symbol_df(raw_records: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subset in sorted(FORMAL_SUBSETS):
        for record in raw_records.get(subset, []):
            task_id = record.get("id")
            paper_name = record.get("paper_name")
            fields = {
                "question": as_list(record.get("questions")),
                "answer": as_list(record.get("answers")),
                "background": as_list(record.get("backgrounds")),
            }
            for source_field, values in fields.items():
                for idx, value in enumerate(values):
                    for symbol, count in Counter(extract_latex_terms(value)).items():
                        rows.append(
                            {
                                "subset": subset,
                                "task_id": task_id,
                                "paper_name": paper_name,
                                "subtask_idx": idx,
                                "source_field": source_field,
                                "symbol": symbol,
                                "count": count,
                            }
                        )
    return pd.DataFrame(rows)


def extract_memory_units(raw_records: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Extract heuristic memory units for all subsets."""
    rows: list[dict[str, Any]] = []

    def add(
        subset: str,
        task_id: Any,
        subtask_idx: int | None,
        memory_type: str,
        memory_value: Any,
        source_field: str,
    ) -> None:
        value = to_text(memory_value, max_chars=500).strip()
        if not value:
            return
        rows.append(
            {
                "subset": subset,
                "task_id": task_id,
                "subtask_idx": -1 if subtask_idx is None else subtask_idx,
                "memory_type": memory_type,
                "memory_value": value,
                "source_field": source_field,
                "estimated_tokens": estimate_tokens(value),
            }
        )

    for subset, records in raw_records.items():
        for fallback_id, record in enumerate(records):
            task_id = _row_id(record, fallback_id)
            if subset == "bundled_shopping":
                for idx, answer in enumerate(as_list(record.get("answers"))):
                    if isinstance(answer, dict):
                        add(subset, task_id, idx, "entity_memory", answer.get("target_asin"), "answers.target_asin")
                        for attr in as_list(answer.get("attributes")):
                            add(subset, task_id, idx, "attribute_memory", attr, "answers.attributes")
                for idx, question in enumerate(as_list(record.get("questions"))):
                    for keyword in CONSTRAINT_KEYWORDS:
                        if count_keyword_constraints(question, [keyword]):
                            add(subset, task_id, idx, "preference_constraint_memory", keyword, "questions")

            elif subset == "progressive_search":
                for idx, answer in enumerate(as_list(record.get("answers"))):
                    add(subset, task_id, idx, "intermediate_result_memory", answer, "answers")
                for idx, question in enumerate(as_list(record.get("questions"))):
                    for keyword in CONSTRAINT_KEYWORDS:
                        if count_keyword_constraints(question, [keyword]):
                            add(subset, task_id, idx, "preference_constraint_memory", keyword, "questions")

            elif subset == "group_travel_planner":
                add(subset, task_id, None, "entity_memory", _base_person_name(record), "base_person.name")
                for idx, question in enumerate(as_list(record.get("questions"))):
                    for name in _extract_person_names_from_questions([question]):
                        add(subset, task_id, idx, "entity_memory", name, "questions")
                    for keyword in TRAVEL_KEYWORDS:
                        if count_keyword_constraints(question, [keyword]):
                            add(subset, task_id, idx, "preference_constraint_memory", keyword, "questions")
                for _, edge in extract_travel_edges({"group_travel_planner": [record]}).iterrows():
                    relation = f"{edge['source_person']} -> {edge['target_person']}"
                    add(subset, task_id, int(edge["subtask_idx"]), "relational_memory", relation, "questions")

            elif subset in FORMAL_SUBSETS:
                add(subset, task_id, None, "entity_memory", record.get("paper_name"), "paper_name")
                for idx, background in enumerate(as_list(record.get("backgrounds"))):
                    add(subset, task_id, idx, "background_definition_memory", background, "backgrounds")
                    for term in extract_latex_terms(background, top_k=20):
                        add(subset, task_id, idx, "entity_memory", term, "backgrounds")
                for idx, answer in enumerate(as_list(record.get("answers"))):
                    add(subset, task_id, idx, "intermediate_result_memory", answer, "answers")
                for idx, question in enumerate(as_list(record.get("questions"))):
                    for term in extract_latex_terms(question, top_k=20):
                        add(subset, task_id, idx, "entity_memory", term, "questions")

    return pd.DataFrame(rows)


def compute_difficulty_score(
    task_df: pd.DataFrame,
    subtask_df: pd.DataFrame | None = None,
    raw_records: dict[str, list[dict[str, Any]]] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute a configurable proxy difficulty score.

    This is not an official MemoryArena metric. It is a normalized heuristic
    designed to rank memory pressure by observable task features.
    """
    del raw_records
    if weights is None:
        weights = {
            "num_questions": 1.2,
            "total_question_tokens": 1.0,
            "total_answer_tokens": 0.8,
            "total_background_tokens": 0.8,
            "answer_structure_complexity": 0.7,
            "constraint_keyword_count": 1.0,
            "entity_or_attribute_count": 0.9,
            "dependency_reference_count": 1.1,
        }
    scored = task_df.copy()
    if "constraint_keyword_count" not in scored and subtask_df is not None:
        constraint_counts = (
            subtask_df.groupby(["subset", "task_id"])["question_keyword_constraint_count"].sum().reset_index()
        )
        scored = scored.merge(
            constraint_counts,
            left_on=["subset", "id"],
            right_on=["subset", "task_id"],
            how="left",
        )
        scored["constraint_keyword_count"] = scored["question_keyword_constraint_count"].fillna(0)
    for column in weights:
        if column not in scored:
            scored[column] = 0.0
        scored[column] = pd.to_numeric(scored[column], errors="coerce").fillna(0.0)

    components = scored[["subset", "id"]].copy()
    score = np.zeros(len(scored), dtype=float)
    total_weight = sum(abs(weight) for weight in weights.values()) or 1.0
    for column, weight in weights.items():
        values = scored[column].astype(float)
        minimum = float(values.min())
        maximum = float(values.max())
        if math.isclose(maximum, minimum):
            normalized = pd.Series(np.zeros(len(values)), index=values.index)
        else:
            normalized = (values - minimum) / (maximum - minimum)
        components[f"{column}_norm"] = normalized
        score += normalized.to_numpy() * weight
    scored["proxy_difficulty_score"] = score / total_weight
    components["proxy_difficulty_score"] = scored["proxy_difficulty_score"]
    return scored, components


def make_subset_summary(task_df: pd.DataFrame, subtask_df: pd.DataFrame) -> pd.DataFrame:
    task_summary = (
        task_df.groupby("subset")
        .agg(
            num_records=("id", "count"),
            invalid_schema_count=("schema_valid", lambda values: int((~values.astype(bool)).sum())),
            avg_num_questions=("num_questions", "mean"),
            median_total_tokens=("total_tokens_est", "median"),
            avg_proxy_difficulty=("proxy_difficulty_score", "mean"),
        )
        .reset_index()
    )
    if subtask_df.empty:
        return task_summary
    subtask_summary = (
        subtask_df.groupby("subset")
        .agg(
            num_subtasks=("subtask_idx", "count"),
            avg_question_tokens=("question_tokens", "mean"),
            avg_answer_tokens=("answer_tokens", "mean"),
            avg_background_tokens=("background_tokens", "mean"),
        )
        .reset_index()
    )
    return task_summary.merge(subtask_summary, on="subset", how="left")


def field_availability_by_subset(
    raw_records: dict[str, list[dict[str, Any]]],
    fields: Iterable[str] = (
        "questions",
        "answers",
        "backgrounds",
        "base_person",
        "paper_name",
        "category",
    ),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subset, records in raw_records.items():
        for field in fields:
            available = sum(1 for record in records if not is_missing_value(record.get(field)))
            rows.append(
                {
                    "subset": subset,
                    "field": field,
                    "availability_rate": available / len(records) if records else 0.0,
                    "available_count": available,
                    "record_count": len(records),
                }
            )
    return pd.DataFrame(rows)


def ensure_output_dirs(base_dir: str | Path) -> tuple[Path, Path, Path]:
    base_path = Path(base_dir)
    figures_dir = base_path / "figures"
    tables_dir = base_path / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return base_path, figures_dir, tables_dir


def save_current_figure(path: str | Path, dpi: int = 160) -> None:
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_bar_counts(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    path: str | Path,
    rotation: int = 30,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data[x].astype(str), data[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=rotation)
    save_current_figure(path)


def plot_boxplot_by_group(
    data: pd.DataFrame,
    value: str,
    group: str,
    title: str,
    path: str | Path,
    rotation: int = 30,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = [frame[value].dropna().values for _, frame in data.groupby(group)]
    labels = [str(label) for label, _ in data.groupby(group)]
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(group)
    ax.set_ylabel(value)
    ax.tick_params(axis="x", rotation=rotation)
    save_current_figure(path)


def plot_hist_by_group(
    data: pd.DataFrame,
    value: str,
    group: str,
    title: str,
    path: str | Path,
    bins: int = 30,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, frame in data.groupby(group):
        ax.hist(frame[value].dropna(), bins=bins, alpha=0.45, label=str(label))
    ax.set_title(title)
    ax.set_xlabel(value)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    save_current_figure(path)


def plot_line(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str,
    path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, frame in data.groupby(group):
        ordered = frame.sort_values(x)
        ax.plot(ordered[x], ordered[y], marker="o", label=str(label))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(fontsize=8)
    save_current_figure(path)


def write_eda_report(
    path: str | Path,
    configs: list[str],
    subset_summary: pd.DataFrame,
    task_df: pd.DataFrame,
    memory_units_df: pd.DataFrame,
) -> None:
    """Write a compact Markdown report from generated EDA tables."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = subset_summary.set_index("subset")["num_records"].to_dict() if not subset_summary.empty else {}
    hardest_subset = "n/a"
    if "proxy_difficulty_score" in task_df and not task_df.empty:
        hardest_subset = (
            task_df.groupby("subset")["proxy_difficulty_score"].mean().sort_values(ascending=False).index[0]
        )
    memory_mix = (
        memory_units_df.groupby(["subset", "memory_type"]).size().reset_index(name="count")
        if not memory_units_df.empty
        else pd.DataFrame(columns=["subset", "memory_type", "count"])
    )
    lines = [
        "# MemoryArena EDA Report",
        "",
        "## Dataset Configs Loaded",
        "",
    ]
    lines.extend(f"- `{config}`" for config in configs)
    lines.extend(["", "## Record Counts", ""])
    lines.extend(f"- `{subset}`: {count}" for subset, count in counts.items())
    lines.extend(
        [
            "",
            "## Data Quality Findings",
            "",
            "- Schema validity is measured as `len(questions) == len(answers)` per task.",
            "- Missingness and field availability are exported in `subset_summary.csv` and notebook tables.",
            "- Length outliers are identified from token and character statistics, not paper constants.",
            "",
            "## Memory Pressure Notes",
            "",
            "- `bundled_shopping`: pressure comes from structured product choices, ASINs, attributes, and compatibility constraints.",
            "- `progressive_search`: pressure comes from accumulating constraints and reusing intermediate answers.",
            "- `group_travel_planner`: pressure comes from person-to-person references and shared itinerary constraints.",
            "- `formal_reasoning_math` and `formal_reasoning_phys`: pressure comes from definitions, notation reuse, and intermediate symbolic results.",
            "",
            "## Proxy Difficulty",
            "",
            f"- Highest average proxy difficulty subset: `{hardest_subset}`.",
            "- This score is a heuristic, not an official MemoryArena metric.",
            "",
            "## Suggested Memory Mechanisms",
            "",
            "- `bundled_shopping`: structured key-value and attribute memory.",
            "- `progressive_search`: constraint accumulation memory.",
            "- `group_travel_planner`: entity-relation graph memory.",
            "- `formal_reasoning_*`: symbolic, definition, and intermediate lemma memory.",
            "",
            "## Memory Unit Mix",
            "",
        ]
    )
    if memory_mix.empty:
        lines.append("- No memory units extracted.")
    else:
        for _, row in memory_mix.sort_values(["subset", "memory_type"]).iterrows():
            lines.append(f"- `{row['subset']}` / `{row['memory_type']}`: {int(row['count'])}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
