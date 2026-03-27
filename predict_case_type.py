#!/usr/bin/env python3
"""Predict case types from case JSON input.

This script loads the trained case type artifacts from:
  models/case_type/

Input JSON can be one of:
1) A single case object
2) A list of case objects
3) An object with a top-level "cases" list

Each case must provide:
- {"documents": [{"id": "...", "title": "...", "text": "..."}, ...]}

Output JSON format:
{
  "task": "case_type",
  "predictions": [
    {"case_id": "123", "predicted_labels": ["LabelA", "LabelB"]}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models" / "case_type"


class InputFormatError(ValueError):
    """Raised when the input JSON structure is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict case type labels from case JSON input."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="-",
        help="Path to input JSON file. Use '-' for stdin (default).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="Path to output JSON file. Use '-' for stdout (default).",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing case_type_model.pkl, tfidf_vectorizer.pkl, and mlb.pkl.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print output JSON.",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    if path == "-":
        raise ValueError("Use load_json_stdin() when input path is '-'.")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_stdin() -> Any:
    import sys

    return json.load(sys.stdin)


def dump_json(data: Dict[str, Any], path: str, pretty: bool) -> None:
    import sys

    if pretty:
        payload = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    if path == "-":
        sys.stdout.write(payload + "\n")
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)
        f.write("\n")


def normalize_cases(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict) and "cases" in raw:
        cases = raw["cases"]
        if not isinstance(cases, list):
            raise InputFormatError("If provided, 'cases' must be a list.")
        return [coerce_case_obj(c, idx) for idx, c in enumerate(cases)]

    if isinstance(raw, list):
        return [coerce_case_obj(c, idx) for idx, c in enumerate(raw)]

    if isinstance(raw, dict):
        return [coerce_case_obj(raw, 0)]

    raise InputFormatError(
        "Input JSON must be a case object, a list of case objects, or an object with 'cases'."
    )


def coerce_case_obj(case: Any, idx: int) -> Dict[str, Any]:
    if not isinstance(case, dict):
        raise InputFormatError(f"Case at index {idx} must be a JSON object.")
    return case


def get_validated_documents(case: Dict[str, Any], case_idx: int) -> List[Dict[str, Any]]:
    docs = case.get("documents")
    if not isinstance(docs, list):
        raise InputFormatError(f"Case at index {case_idx} must include a 'documents' list.")

    for doc_idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise InputFormatError(
                f"Case at index {case_idx}, document at index {doc_idx} must be a JSON object."
            )
        if "id" not in doc:
            raise InputFormatError(
                f"Case at index {case_idx}, document at index {doc_idx} is missing required field 'id'."
            )

    return docs


def case_to_text(case: Dict[str, Any], case_idx: int) -> str:
    # Keep aggregation behavior aligned with training:
    # "{title} {text}" per document, joined with single spaces.
    docs = get_validated_documents(case, case_idx)

    chunks: List[str] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        title = doc.get("title", "")
        text = doc.get("text", "")
        if not isinstance(title, str):
            title = ""
        if not isinstance(text, str):
            text = ""

        merged = f"{title} {text}".strip()
        if merged:
            chunks.append(merged)

    return " ".join(chunks)


def prepare_texts(cases: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    case_ids: List[str] = []

    for idx, case in enumerate(cases):
        cid = case.get("case_id", idx)
        case_id = str(cid)
        text = case_to_text(case, idx)

        texts.append(text)
        case_ids.append(case_id)

    return texts, case_ids


def load_artifacts(model_dir: Path):
    model_path = model_dir / "case_type_model.pkl"
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"
    mlb_path = model_dir / "mlb.pkl"

    missing = [
        str(p)
        for p in [model_path, vectorizer_path, mlb_path]
        if not p.exists()
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing required model artifacts: {joined}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)

    return model, vectorizer, mlb


def predict_cases(cases: List[Dict[str, Any]], model_dir: Path) -> Dict[str, Any]:
    model, vectorizer, mlb = load_artifacts(model_dir)
    texts, case_ids = prepare_texts(cases)

    X = vectorizer.transform(texts)
    y_pred = model.predict(X)
    labels_per_case = mlb.inverse_transform(y_pred)

    predictions = []
    for case_id, labels in zip(case_ids, labels_per_case):
        predictions.append(
            {
                "case_id": case_id,
                "predicted_labels": list(labels),
            }
        )

    return {
        "task": "case_type",
        "predictions": predictions,
    }


def main() -> None:
    import sys

    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()

    try:
        raw_input = load_json_stdin() if args.input == "-" else load_json(args.input)
        cases = normalize_cases(raw_input)
        output = predict_cases(cases, model_dir)
        dump_json(output, args.output, args.pretty)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
