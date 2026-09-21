#!/usr/bin/env python3
"""Shared malformed-response rules for compact API prediction artifacts."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from api_batch_common import BatchIntegrityError


MISSING_PARSE_ERROR_VALUES = {"", "nan", "None"}


def _parse_error_mask(series: pd.Series) -> pd.Series:
    return ~series.astype(str).str.strip().isin(MISSING_PARSE_ERROR_VALUES)


def prediction_malformed_masks(
    predictions: pd.DataFrame,
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[pd.Series, pd.Series]:
    """Return whole-response malformed and schema-invalid masks.

    Any invalid required label marks the entire logical response malformed.  In
    particular, one bad value in a multi-binary response does not leave its
    other decisions eligible to be scored as if the response had parsed.
    """
    required = {"task", "parse_error"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise BatchIntegrityError(
            f"predictions are missing malformed-validation columns: {missing}"
        )
    task_by_name = {str(task["name"]): task for task in tasks}
    observed_tasks = set(predictions["task"].astype(str))
    unknown = sorted(observed_tasks - set(task_by_name))
    if unknown:
        raise BatchIntegrityError(
            f"predictions contain tasks without a frozen schema: {unknown}"
        )

    schema_invalid = pd.Series(False, index=predictions.index, dtype=bool)
    for task_name in sorted(observed_tasks):
        task = task_by_name[task_name]
        task_rows = predictions["task"].astype(str) == task_name
        label_kind = str(task["label_kind"])
        if label_kind == "multi_binary":
            valid = pd.Series(True, index=predictions.index, dtype=bool)
            for label in task["labels"]:
                column = f"pred_{label}"
                if column not in predictions:
                    valid.loc[task_rows] = False
                else:
                    values = pd.to_numeric(predictions[column], errors="coerce")
                    valid.loc[task_rows] &= values.loc[task_rows].isin([0, 1])
            schema_invalid.loc[task_rows] = ~valid.loc[task_rows]
        elif label_kind == "binary":
            column = f"pred_{task['label_key']}"
            if column not in predictions:
                schema_invalid.loc[task_rows] = True
            else:
                values = pd.to_numeric(predictions[column], errors="coerce")
                schema_invalid.loc[task_rows] = ~values.loc[task_rows].isin([0, 1])
        elif label_kind == "categorical":
            column = f"pred_{task['label_key']}"
            if column not in predictions:
                schema_invalid.loc[task_rows] = True
            else:
                allowed = {str(label) for label in task["labels"]}
                schema_invalid.loc[task_rows] = ~predictions.loc[
                    task_rows, column
                ].astype(str).isin(allowed)
        else:
            raise BatchIntegrityError(
                f"{task_name}: unsupported frozen label kind {label_kind!r}"
            )

    parse_error = _parse_error_mask(predictions["parse_error"])
    return parse_error | schema_invalid, schema_invalid
