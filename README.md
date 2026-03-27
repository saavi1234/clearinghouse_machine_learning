# clearinghouse_machine_learning

Inference scripts for 5 trained ML classifiers:

1. plaintiff_type (case-level, multi-label)
2. case_type (case-level, multi-label)
3. defendant_type (case-level, multi-label)
4. document_type (document-level, single-label)
5. party_type (document-level, multi-label)

## Repository Layout

- [models/case_type](models/case_type)
- [models/defendant_type](models/defendant_type)
- [models/document_type](models/document_type)
- [models/party_type](models/party_type)
- [models/plaintiff_type](models/plaintiff_type)

Prediction scripts:

- [predict_plaintiff_type.py](predict_plaintiff_type.py)
- [predict_case_type.py](predict_case_type.py)
- [predict_defendant_type.py](predict_defendant_type.py)
- [predict_document_type.py](predict_document_type.py)
- [predict_party_type.py](predict_party_type.py)

Sample input:

- [sample_cases.json](sample_cases.json)

## Setup

From repo root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Input Format

All predictors accept JSON in one of these forms:

1. A single case object
2. A list of case objects
3. An object with a top-level cases list

Each case must include:

- case_id (recommended)
- documents: list of document objects

Each document must include:

- id
- title (optional)
- text (optional)

Aggregated case-level text input is not supported.

## Output Format

Case-level predictors output:

```json
{
	"task": "plaintiff_type",
	"predictions": [
		{
			"case_id": "sample-001",
			"predicted_labels": ["..."]
		}
	]
}
```

Document-level predictors output:

```json
{
	"task": "document_type",
	"predictions": [
		{
			"case_id": "sample-001",
			"document_id": "doc-001",
			"predicted_label": "..."
		}
	]
}
```

For party_type, the field is predicted_labels (multi-label) instead of predicted_label.

## Sample Usage (Using sample_cases.json)

Run from repo root.

### 1) Plaintiff Type

```bash
.venv/bin/python predict_plaintiff_type.py -i sample_cases.json --pretty
```

### 2) Case Type

```bash
.venv/bin/python predict_case_type.py -i sample_cases.json --pretty
```

### 3) Defendant Type

```bash
.venv/bin/python predict_defendant_type.py -i sample_cases.json --pretty
```

### 4) Document Type

```bash
.venv/bin/python predict_document_type.py -i sample_cases.json --pretty
```

### 5) Party Type

```bash
.venv/bin/python predict_party_type.py -i sample_cases.json --pretty
```

## Write Output to Files

Example:

```bash
.venv/bin/python predict_case_type.py -i sample_cases.json -o case_type_predictions.json --pretty
```

## Stdin/Stdout Example

```bash
cat sample_cases.json | .venv/bin/python predict_document_type.py --pretty
```