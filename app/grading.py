"""Utilities for grading chatbot responses with a dedicated Qwen judge."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama

GRADER_SYSTEM_PROMPT = (
    "You are an impartial grading assistant for a course-planning chatbot. "
    "Follow the evaluation rubric exactly and respond with a single JSON object "
    "containing only the keys accuracy, relevance, and coherence. Round each "
    "score to one decimal place and do not include any extra commentary, "
    "markdown, or prose."
)

GRADER_USER_TEMPLATE = (
    "You are an impartial grading assistant for a course-planning chatbot. "
    "Your task is to evaluate the chatbot’s response to a given academic query.\n\n"
    "QUESTION:\n{question}\n\n"
    "GROUND TRUTH:\n{ground_truth}\n\n"
    "PREDICTED ANSWER:\n{answer}\n\n"
    "---\n\n"
    "### Evaluation Criteria (0–1 for each)\n"
    "1. **Accuracy (0–1):** How correctly does the chatbot understand the user and provide the right information? You can measure this through intent recognition or by comparing the response to the ground truth.\n\n"
    "2. **Relevance (0–1):** Does the response directly answer the question or address the user's needs?\n\n"
    "3. **Fluency and Coherence (0–1):** Is the response grammatically correct, easy to understand, and logically structured?\n\n"
    "---\n\n"
    "Each score must be between 0 and 1, using increments of 0.1.\n\n"
    "### Output Format\n"
    "Return only a single JSON object with numeric scores rounded to one decimal place, formatted as follows:\n\n"
    "{{\"accuracy\": {{score}}, \"relevance\": {{score}}, \"coherence\": {{score}}}}"
)

EXPECTED_SCORE_KEYS = ("accuracy", "relevance", "coherence")

SCORE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(accuracy|relevance|coherence)(?![A-Za-z0-9_])\s*[:=]\s*"
    r"(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass
class GradePayload:
    """Structured result returned after grading."""

    evaluation: Dict[str, Any]
    developer: Optional[Dict[str, Any]]


def _safe_json(value: Any) -> Any:
    """Coerce complex structures into JSON-serialisable primitives."""

    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse a JSON object from the model output."""

    candidate = text.strip()
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _parse_scores(text: str) -> Tuple[Dict[str, float], Optional[str], Optional[Dict[str, Any]]]:
    """Parse grader scores from the response text."""

    parsed_json = _extract_json_object(text)
    raw_values: Dict[str, Any] = {}
    if parsed_json:
        raw_values = {key.lower(): value for key, value in parsed_json.items()}
    else:
        for match in SCORE_PATTERN.finditer(text):
            key = match.group(1).lower()
            raw_values[key] = match.group(2)

    scores: Dict[str, float] = {}
    missing_keys = []
    invalid_keys = []
    for key in EXPECTED_SCORE_KEYS:
        value = raw_values.get(key)
        if value is None:
            missing_keys.append(key)
            continue
        try:
            numeric = round(float(value), 1)
        except (TypeError, ValueError):
            invalid_keys.append(key)
            continue
        numeric = max(0.0, min(1.0, numeric))
        scores[key] = numeric

    error_parts = []
    if invalid_keys:
        error_parts.append(
            "Invalid numeric value for: " + ", ".join(sorted(invalid_keys))
        )
    if missing_keys:
        error_parts.append("Missing scores for: " + ", ".join(sorted(missing_keys)))
    if not scores:
        error_parts.append("Grader response did not contain usable scores.")

    error_message = " ".join(error_parts) if error_parts else None
    return scores, error_message, parsed_json


class ResponseGrader:
    """LLM-backed grader that evaluates chatbot answers."""

    def __init__(self) -> None:
        self._llm = ChatOllama(
            model="qwen3:14b",
            temperature=0.0,
            num_predict=-1,
            reasoning=True,
            validate_model_on_init=True,
        )

    def grade(
        self,
        *,
        question: str,
        ground_truth: str,
        answer: str,
        developer_view: bool = False,
    ) -> GradePayload:
        """Return grader scores and optional developer tracing information."""

        ground_truth_text = ground_truth.strip() or "[ground truth not provided]"
        user_prompt = GRADER_USER_TEMPLATE.format(
            question=question.strip(),
            ground_truth=ground_truth_text,
            answer=answer.strip() or "[no answer provided]",
        )

        messages = [
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        evaluation: Dict[str, Any] = {"grader_prompt": user_prompt}
        developer_payload: Optional[Dict[str, Any]] = None

        try:
            response = self._llm.invoke(messages)
        except Exception as exc:  # pragma: no cover - defensive guard for runtime issues
            evaluation["error"] = str(exc)
            if developer_view:
                developer_payload = {
                    "messages": [
                        {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "error": str(exc),
                }
            return GradePayload(evaluation=evaluation, developer=developer_payload)

        raw_text = response.content.strip()
        raw_additional_kwargs = getattr(response, "additional_kwargs", {})
        raw_reasoning: Any | None = None
        if isinstance(raw_additional_kwargs, dict):
            for key in ("reasoning", "reasoning_content", "thoughts"):
                if key in raw_additional_kwargs:
                    raw_reasoning = raw_additional_kwargs[key]
                    break

        scores, score_error, parsed_json = _parse_scores(raw_text)
        if scores:
            evaluation["scores"] = scores
            evaluation["total"] = round(sum(scores.values()), 1)

        if score_error and "error" not in evaluation:
            evaluation["error"] = score_error

        if raw_reasoning is not None:
            evaluation["grader_reasoning"] = _safe_json(raw_reasoning)

        if developer_view:
            developer_payload = {
                "messages": [
                    {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "raw_response": raw_text,
                "parsed_scores": scores,
                "parser": {
                    "raw_json": _safe_json(parsed_json) if parsed_json is not None else None,
                    "used_regex_fallback": parsed_json is None,
                    "error": score_error,
                },
                "response_metadata": _safe_json(getattr(response, "response_metadata", {})),
                "additional_kwargs": _safe_json(raw_additional_kwargs),
            }
            reasoning_traces = None
            additional = developer_payload["additional_kwargs"]
            if isinstance(additional, dict):
                for key in ("reasoning", "reasoning_content", "thoughts"):
                    if key in additional:
                        reasoning_traces = additional[key]
                        break
            if reasoning_traces is not None:
                developer_payload["reasoning_traces"] = reasoning_traces

        if "error" not in evaluation and not scores:
            evaluation["error"] = "Grader response did not contain usable scores."

        return GradePayload(evaluation=evaluation, developer=developer_payload)
