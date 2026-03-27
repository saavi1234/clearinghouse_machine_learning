#!/usr/bin/env python3
"""Predict document types from case JSON input.

This script loads the trained document type artifacts from:
  models/document_type/

Input JSON can be one of:
1) A single case object
2) A list of case objects
3) An object with a top-level "cases" list

Each case is expected to contain a "documents" list. Each document must include:
- "id"
- "title"
- "text"

Output JSON format:
{
  "task": "document_type",
  "predictions": [
    {
      "case_id": "123",
      "document_id": "abc",
      "predicted_label": "Complaint"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.preprocessing import normalize


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models" / "document_type"


class InputFormatError(ValueError):
    """Raised when the input JSON structure is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict document type labels from case JSON input."
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
        help="Directory containing linearsvc_model.pkl and tfidf_vectorizer.pkl.",
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


def clean_text(text: str) -> str:
    # Match training-time cleaning behavior as closely as possible.
    text = str(text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", " ", text)
    text = re.sub(r"[^\w\s\.\,\!\?\-\(\)\:\;]", " ", text)
    text = text.lower().strip()
    return text if text else " "


def extract_documents(cases: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, str]]]:
    texts: List[str] = []
    metas: List[Dict[str, str]] = []

    for case_idx, case in enumerate(cases):
        case_id = str(case.get("case_id", case_idx))
        docs = get_validated_documents(case, case_idx)

        for doc in docs:
            if not isinstance(doc, dict):
                continue

            doc_id = str(doc["id"])
            title = doc.get("title", "")
            body = doc.get("text", "")

            if not isinstance(title, str):
                title = ""
            if not isinstance(body, str):
                body = ""

            combined = (title + "\n" + body).strip()
            cleaned = clean_text(combined)

            texts.append(cleaned)
            metas.append({"case_id": case_id, "document_id": doc_id})

    return texts, metas


def load_artifacts(model_dir: Path):
    model_path = model_dir / "linearsvc_model.pkl"
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"

    missing = [str(p) for p in [model_path, vectorizer_path] if not p.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing required model artifacts: {joined}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_documents(cases: List[Dict[str, Any]], model_dir: Path) -> Dict[str, Any]:
    model, vectorizer = load_artifacts(model_dir)
    texts, metas = extract_documents(cases)

    if not texts:
        return {"task": "document_type", "predictions": []}

    X = vectorizer.transform(texts)
    X = normalize(X, norm="l2")
    preds = model.predict(X)

    predictions = []
    for meta, pred in zip(metas, preds):
        predictions.append(
            {
                "case_id": meta["case_id"],
                "document_id": meta["document_id"],
                "predicted_label": str(pred),
            }
        )

    return {
        "task": "document_type",
        "predictions": predictions,
    }


def main() -> None:
    import sys

    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()

    try:
        raw_input = load_json_stdin() if args.input == "-" else load_json(args.input)
        cases = normalize_cases(raw_input)
        output = predict_documents(cases, model_dir)
        dump_json(output, args.output, args.pretty)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
