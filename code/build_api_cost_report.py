#!/usr/bin/env python3
"""Aggregate exact per-checkpoint API preflights into one approval report."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import yaml

from api_batch_common import BatchIntegrityError, file_sha256


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.000001")))


def build_aggregate_report(
    availability_manifest_path: str | Path,
    preflight_root: str | Path,
    *,
    expected_request_count: int | None = None,
    hard_budget_usd: str | Decimal | None = None,
    drop_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    availability_path = Path(availability_manifest_path)
    availability = yaml.safe_load(availability_path.read_text())
    if not isinstance(availability, dict) or availability.get("schema_version") != 1:
        raise BatchIntegrityError("availability manifest schema_version must be 1")
    checkpoints = availability.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise BatchIntegrityError("availability manifest checkpoints must be a mapping")

    expected_request_count = int(
        expected_request_count
        if expected_request_count is not None
        else availability.get("expected_request_count", 8793)
    )
    pilot_size = int(availability.get("pilot_size", 0) or 0)
    if expected_request_count < 1:
        raise BatchIntegrityError("expected_request_count must be positive")
    if pilot_size and not 0 < pilot_size < expected_request_count:
        raise BatchIntegrityError(
            "pilot_size must be positive and smaller than expected_request_count"
        )
    if hard_budget_usd is None:
        hard_budget_usd = availability.get("hard_budget_usd")
    budget = Decimal(str(hard_budget_usd)) if hard_budget_usd is not None else None
    if budget is not None and budget < 0:
        raise BatchIntegrityError("hard_budget_usd must be non-negative")
    if drop_order is None:
        raw_drop_order = availability.get("drop_policy", [])
        if not isinstance(raw_drop_order, list):
            raise BatchIntegrityError("drop_policy must be a list of checkpoint ids")
        drop_order = [str(value) for value in raw_drop_order]
    if len(set(drop_order)) != len(drop_order):
        raise BatchIntegrityError("drop_policy contains duplicate checkpoint ids")

    root = Path(preflight_root)
    priced_models: list[dict[str, Any]] = []
    missing_reports: list[str] = []
    for checkpoint_id, state in checkpoints.items():
        if not bool(state.get("preflight_required")):
            continue
        checkpoint_root = root / checkpoint_id
        staged = bool(state.get("staged_preflight_required"))
        if staged:
            if not pilot_size:
                raise BatchIntegrityError(
                    f"{checkpoint_id}: staged preflight requires pilot_size"
                )
            paths = [
                ("pilot", checkpoint_root / "pilot" / "preflight.json", pilot_size),
                (
                    "remainder",
                    checkpoint_root / "remainder" / "preflight.json",
                    expected_request_count - pilot_size,
                ),
            ]
        else:
            paths = [("", checkpoint_root / "preflight.json", expected_request_count)]
        missing = [stage for stage, path, _count in paths if not path.is_file()]
        if missing:
            missing_reports.extend(
                f"{checkpoint_id}:{stage}" if stage else checkpoint_id
                for stage in missing
            )
            continue
        reports: list[tuple[str, Path, dict[str, Any]]] = []
        for stage, report_path, expected_stage_count in paths:
            report = json.loads(report_path.read_text())
            expected = {
                "schema_version": 1,
                "requested_model": state.get("model_id"),
                "request_count": expected_stage_count,
            }
            if staged:
                expected.update(
                    {
                        "request_stage": stage,
                        "planned_request_count": expected_request_count,
                        "pilot_size": pilot_size,
                    }
                )
            for field, value in expected.items():
                if report.get(field) != value:
                    raise BatchIntegrityError(
                        f"{checkpoint_id}: {stage or 'full'} preflight {field} "
                        "does not match the compact API roster"
                    )
            provider = str(report.get("provider") or "")
            if provider.lower() != str(state.get("provider") or "").lower():
                raise BatchIntegrityError(
                    f"{checkpoint_id}: preflight provider mismatch"
                )
            reports.append((stage, report_path, report))

        identity_fields = [
            "provider",
            "requested_model",
            "model_identity_sha256",
            "panel_id",
            "panel_sha256",
            "task_item_keys_sha256",
            "pricing_manifest_sha256",
        ]
        for field in identity_fields:
            values = {str(report.get(field)) for _stage, _path, report in reports}
            if len(values) != 1:
                raise BatchIntegrityError(
                    f"{checkpoint_id}: staged preflights disagree on {field}"
                )

        def sum_int(field: str) -> int:
            return sum(int(report[field]) for _stage, _path, report in reports)

        def sum_money(field: str) -> Decimal:
            return sum(
                (Decimal(str(report[field])) for _stage, _path, report in reports),
                Decimal("0"),
            )

        first_report = reports[0][2]
        preflight_reports = [
            {
                "request_stage": stage or "all",
                "request_count": int(report["request_count"]),
                "estimated_cost_usd": str(report["estimated_total_cost_usd"]),
                "maximum_cost_usd": str(report["maximum_cost_usd"]),
                "path": str(report_path.resolve()),
                "sha256": file_sha256(report_path),
            }
            for stage, report_path, report in reports
        ]
        model_row = {
            "checkpoint_id": checkpoint_id,
            "provider": str(first_report["provider"]),
            "model_id": first_report["requested_model"],
            "model_identity_sha256": first_report.get("model_identity_sha256"),
            "identity_class": state.get("identity_class"),
            "lifecycle_status": state.get("lifecycle_status"),
            "snapshot_release_date": (
                state["snapshot_release_date"].isoformat()
                if hasattr(state.get("snapshot_release_date"), "isoformat")
                else state.get("snapshot_release_date")
            ),
            "request_count": sum_int("request_count"),
            "pilot_request_count": pilot_size if staged else 0,
            "remainder_request_count": (
                expected_request_count - pilot_size if staged else 0
            ),
            "estimated_input_tokens": sum_int("estimated_input_tokens"),
            "maximum_input_tokens": sum_int("maximum_input_tokens"),
            "estimated_output_tokens": sum_int("estimated_output_tokens"),
            "maximum_output_tokens": sum_int("maximum_output_tokens"),
            "estimated_cost_usd": _money(sum_money("estimated_total_cost_usd")),
            "maximum_cost_usd": _money(sum_money("maximum_cost_usd")),
            "preflight_reports": preflight_reports,
        }
        if len(reports) == 1:
            model_row["preflight_report_path"] = preflight_reports[0]["path"]
            model_row["preflight_report_sha256"] = preflight_reports[0]["sha256"]
        priced_models.append(model_row)

    budget_exclusions: list[dict[str, Any]] = []
    models = list(priced_models)
    if not missing_reports and budget is not None:
        for checkpoint_id in drop_order:
            maximum = sum(
                (Decimal(row["maximum_cost_usd"]) for row in models),
                Decimal("0"),
            )
            if maximum <= budget:
                break
            matching = [
                row for row in models if row["checkpoint_id"] == checkpoint_id
            ]
            if len(matching) != 1:
                raise BatchIntegrityError(
                    f"drop_policy checkpoint is not uniquely priced: {checkpoint_id}"
                )
            dropped = matching[0]
            models.remove(dropped)
            budget_exclusions.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "model_id": dropped["model_id"],
                    "maximum_cost_usd": dropped["maximum_cost_usd"],
                    "reason": "deterministic_hard_budget_drop",
                }
            )

    subtotals: dict[str, dict[str, Decimal | int]] = defaultdict(
        lambda: {
            "models": 0,
            "estimated_cost_usd": Decimal("0"),
            "maximum_cost_usd": Decimal("0"),
        }
    )
    for row in models:
        subtotal = subtotals[row["provider"]]
        subtotal["models"] = int(subtotal["models"]) + 1
        subtotal["estimated_cost_usd"] = Decimal(
            subtotal["estimated_cost_usd"]
        ) + Decimal(row["estimated_cost_usd"])
        subtotal["maximum_cost_usd"] = Decimal(
            subtotal["maximum_cost_usd"]
        ) + Decimal(row["maximum_cost_usd"])

    provider_rows = {
        provider: {
            "models": int(values["models"]),
            "estimated_cost_usd": _money(Decimal(values["estimated_cost_usd"])),
            "maximum_cost_usd": _money(Decimal(values["maximum_cost_usd"])),
        }
        for provider, values in sorted(subtotals.items())
    }
    overall_estimated = sum(
        (Decimal(row["estimated_cost_usd"]) for row in provider_rows.values()),
        Decimal("0"),
    )
    overall_maximum = sum(
        (Decimal(row["maximum_cost_usd"]) for row in provider_rows.values()),
        Decimal("0"),
    )
    mutable = [
        checkpoint_id
        for checkpoint_id, state in checkpoints.items()
        if str(state.get("identity_class") or "").startswith("mutable")
    ]
    retired = [
        checkpoint_id
        for checkpoint_id, state in checkpoints.items()
        if state.get("access_status") == "retired"
    ]
    inaccessible = [
        checkpoint_id
        for checkpoint_id, state in checkpoints.items()
        if state.get("access_status") not in {"accessible", "retired"}
    ]
    budget_compliant = (
        None if missing_reports else (budget is None or overall_maximum <= budget)
    )
    expected_model_count = (
        sum(bool(state.get("preflight_required")) for state in checkpoints.values())
        - len(budget_exclusions)
    )
    approval_ready = (
        not missing_reports
        and budget_compliant is True
        and len(models) == expected_model_count
    )
    approval_status = (
        "incomplete_preflight"
        if missing_reports
        else "pending_explicit_user_approval"
    )
    return {
        "schema_version": 1,
        "report_type": availability.get(
            "report_type", "frontier_api_aggregate_cost_approval"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "availability_manifest_path": str(availability_path.resolve()),
        "availability_manifest_sha256": file_sha256(availability_path),
        "model_count": len(models),
        "expected_model_count": expected_model_count,
        "planned_model_count": sum(
            bool(state.get("preflight_required")) for state in checkpoints.values()
        ),
        "expected_request_count_per_model": expected_request_count,
        "pilot_size": pilot_size,
        "models": models,
        "provider_subtotals": provider_rows,
        "overall_estimated_cost_usd": _money(overall_estimated),
        "overall_maximum_cost_usd": _money(overall_maximum),
        "missing_preflight_reports": missing_reports,
        "hard_budget_usd": _money(budget) if budget is not None else None,
        "budget_compliant": budget_compliant,
        "drop_policy": list(drop_order),
        "budget_exclusions": budget_exclusions,
        "retired_checkpoints": retired,
        "inaccessible_checkpoints": inaccessible,
        "mutable_exclusions": mutable,
        "approval_ready": approval_ready,
        "approval_status": approval_status,
        "submission_enabled": False,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    if report["approval_ready"]:
        verdict = (
            "The exact roster is within the hard ceiling. Paid submission remains "
            "disabled until the user approves this file's exact hash and maximum."
        )
        action = "Review the model roster and exact maximum, then create a matching approval record."
    else:
        verdict = (
            "This report is not approval-ready. Missing exact preflights mean the "
            "final maximum is not known, so paid submission remains disabled."
        )
        action = (
            "Complete the missing provider token counts, rebuild this report, and review "
            "the resulting exact maximum before approving any paid inference."
        )
    if report.get("budget_compliant") is None:
        budget_text = "not yet known because exact preflights are missing"
    else:
        budget_text = str(report["budget_compliant"]).lower()
    lines = [
        "# API cost-approval report",
        "",
        verdict,
        "",
        f"Status: `{report['approval_status']}`. Submission is disabled.",
        f"Action required: {action}",
        "",
        (
            f"This report compares {report['planned_model_count']} API checkpoints on "
            f"the same {report['expected_request_count_per_model']:,}-request compact "
            f"benchmark. Each model is split into a fixed {report['pilot_size']}-request "
            "parser pilot and the non-overlapping remainder."
        ),
        "",
        f"Prepared models: {report['model_count']} of {report['expected_model_count']}.",
        f"Estimated total: ${report['overall_estimated_cost_usd']}.",
        f"Maximum total: ${report['overall_maximum_cost_usd']}.",
        (
            f"Hard ceiling: ${report['hard_budget_usd']} "
            f"(compliant: {budget_text})."
            if report.get("hard_budget_usd") is not None
            else "Hard ceiling: not configured."
        ),
        "",
        "| Checkpoint | Provider | Requests | Estimated USD | Maximum USD |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["models"]:
        lines.append(
            f"| {row['checkpoint_id']} | {row['provider']} | {row['request_count']:,} | "
            f"{row['estimated_cost_usd']} | {row['maximum_cost_usd']} |"
        )
    lines.extend(
        [
            "",
            "## Provider subtotals",
            "",
            "| Provider | Models | Estimated USD | Maximum USD |",
            "|---|---:|---:|---:|",
        ]
    )
    for provider, subtotal in report["provider_subtotals"].items():
        lines.append(
            f"| {provider} | {subtotal['models']} | "
            f"{subtotal['estimated_cost_usd']} | {subtotal['maximum_cost_usd']} |"
        )
    if report["missing_preflight_reports"]:
        lines.extend(
            [
                "",
                "Missing preflights: " + ", ".join(report["missing_preflight_reports"]),
            ]
        )
    if report.get("budget_exclusions"):
        lines.extend(["", "## Deterministic budget exclusions", ""])
        for row in report["budget_exclusions"]:
            lines.append(
                f"- `{row['checkpoint_id']}` ({row['model_id']}), "
                f"maximum ${row['maximum_cost_usd']}"
            )
    lines.extend(
        [
            "",
            "## Cost assumptions",
            "",
            "1. Input counts use the exact serialized provider request for each item. OpenAI uses the pinned local tokenizer; Anthropic requires its model-specific token-count endpoint.",
            "2. The maximum assumes every request consumes its full 128-token output cap. The estimate assumes 64 output tokens per request.",
            "3. The report covers Batch API rates effective on the pricing-manifest date. It does not authorize or submit any request.",
            "",
            "Mutable exclusions: "
            + (", ".join(report["mutable_exclusions"]) or "none"),
            "Retired checkpoints: "
            + (", ".join(report["retired_checkpoints"]) or "none"),
            "Other inaccessible checkpoints: "
            + (", ".join(report["inaccessible_checkpoints"]) or "none"),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability-manifest", required=True)
    parser.add_argument("--preflight-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    report = build_aggregate_report(
        args.availability_manifest,
        args.preflight_root,
    )
    _atomic_json(Path(args.output_json), report)
    write_markdown(Path(args.output_markdown), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["approval_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
