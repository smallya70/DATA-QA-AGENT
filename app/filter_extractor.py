from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from app.planning_context import PlanningContext, normalize_name, normalize_text


@dataclass(slots=True)
class FilterCandidate:
    raw_value: Any
    operator: str = "="
    column_hint: str | None = None
    kind: str = "equality"
    confidence: float = 0.5
    source: str = "deterministic"
    reasons: list[str] = field(default_factory=list)


def _column_phrase(column_name: str) -> str:
    return re.escape(normalize_name(column_name).replace("_", " "))


def _clean_comparison_piece(value: str) -> str:
    normalized = normalize_text(value)
    prefix_pattern = r"^(?:compare|comparing|difference|between|show|total|sum|sales|revenue|profit|amount|value|price|quantity|qty)\s+"
    while True:
        updated = re.sub(prefix_pattern, "", normalized)
        if updated == normalized:
            break
        normalized = normalize_text(updated)
    return normalize_text(normalized)


def _trim_filter_value(value: str) -> str:
    normalized = normalize_text(value)
    normalized = re.sub(r"\s+and\s+(?:year|month|quarter)\s+(?:19|20)\d{2}$", "", normalized)
    normalized = re.sub(r"\s+and\s+[a-z0-9][a-z0-9 _/-]*$", "", normalized)
    return normalize_text(normalized)


def _pick_status_like_column(context: PlanningContext, target_sheet: str | None) -> str | None:
    if not target_sheet:
        return None
    role_map = context.semantic_roles.get(target_sheet) or {}
    for preferred_role in ["exception_type", "billing_status", "status"]:
        mapped = str(role_map.get(preferred_role) or "").strip()
        if mapped:
            return mapped
    for column_name in context.columns_by_sheet.get(target_sheet) or []:
        normalized = normalize_name(column_name)
        if {"status", "state", "stage", "exception"} & set(normalized.split("_")):
            return str(column_name)
    return None


def _pick_invoice_status_like_column(context: PlanningContext, target_sheet: str | None) -> str | None:
    if not target_sheet:
        return None
    role_map = context.semantic_roles.get(target_sheet) or {}
    for preferred_role in ["invoice_status", "billing_status", "status"]:
        mapped = str(role_map.get(preferred_role) or "").strip()
        normalized = normalize_name(mapped)
        if mapped and ((preferred_role == "status" and {"invoice", "bill", "billing"} & set(normalized.split("_"))) or preferred_role != "status"):
            return mapped
    for column_name in context.columns_by_sheet.get(target_sheet) or []:
        normalized = normalize_name(column_name)
        tokens = set(normalized.split("_"))
        if "status" in tokens and ({"invoice", "bill", "billing"} & tokens):
            return str(column_name)
    return None


def extract_filters(
    question: str,
    context: PlanningContext,
    *,
    target_sheet: str | None = None,
) -> list[FilterCandidate]:
    q_text = str(question or "").strip().lower()
    candidates: list[FilterCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    columns = context.columns_by_sheet.get(target_sheet or "", []) if target_sheet else []
    reserved_values = {normalize_text(column_name) for column_name in columns}
    if target_sheet:
        reserved_values.update(normalize_text(role_name) for role_name in (context.semantic_roles.get(target_sheet) or {}).keys())
    if not columns:
        columns = [column for cols in context.columns_by_sheet.values() for column in cols]

    for column_name in columns:
        phrase = _column_phrase(column_name)
        patterns = [
            (rf"\b{phrase}\b\s*(=|is|equals)\s*([a-z0-9][a-z0-9 _/-]*)", "equality", 0.94),
            (rf"\b{phrase}\b\s+after\s+(\d{{4}})", "range", 0.90),
            (rf"\b{phrase}\b\s+before\s+(\d{{4}})", "range", 0.90),
            (rf"\b{phrase}\b\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)", "range", 0.92),
            (rf"\bin\s+{phrase}\b\s*([a-z0-9][a-z0-9 _/-]*)", "equality", 0.84),
            (rf"\bfor\s+{phrase}\b\s*([a-z0-9][a-z0-9 _/-]*)", "equality", 0.82),
        ]
        for pattern, kind, confidence in patterns:
            for match in re.finditer(pattern, q_text):
                groups = match.groups()
                if not groups:
                    continue
                if kind == "range" and len(groups) == 1:
                    raw_value = groups[0]
                    operator = ">"
                elif kind == "range" and len(groups) >= 2:
                    operator = groups[0]
                    raw_value = groups[1]
                elif len(groups) >= 2:
                    operator = groups[0] if groups[0] in {"=", "is", "equals", ">", "<", ">=", "<="} else "="
                    raw_value = groups[1] if operator != "=" or len(groups) > 1 else groups[0]
                else:
                    operator = "="
                    raw_value = groups[0]
                normalized_value = _trim_filter_value(raw_value)
                key = (normalize_name(column_name), str(operator), normalized_value)
                if not normalized_value or key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    FilterCandidate(
                        raw_value=normalized_value,
                        operator="=" if operator in {"is", "equals"} else operator,
                        column_hint=column_name,
                        kind=kind,
                        confidence=confidence,
                        source="explicit_column_filter",
                        reasons=[f"Matched explicit filter syntax for column '{column_name}'."],
                    )
                )

    year_match = re.search(r"\b(after|before|since|until)\s+(19|20)\d{2}\b", q_text)
    if year_match:
        operator = ">=" if year_match.group(1) in {"after", "since"} else "<="
        candidates.append(
            FilterCandidate(
                raw_value=year_match.group(0).split()[-1],
                operator=operator,
                column_hint="year",
                kind="year",
                confidence=0.86,
                source="year_phrase",
                reasons=["Matched year range phrase."],
            )
        )

    exact_year_matches = re.finditer(r"\byear\b\s*(?:=|is)?\s*((?:19|20)\d{2})\b", q_text)
    for match in exact_year_matches:
        year_value = normalize_text(match.group(1))
        key = ("year", "=", year_value)
        if year_value and key not in seen:
            seen.add(key)
            candidates.append(
                FilterCandidate(
                    raw_value=year_value,
                    operator="=",
                    column_hint="year",
                    kind="year",
                    confidence=0.9,
                    source="year_equality_phrase",
                    reasons=["Matched explicit year equality phrase."],
                )
            )

    region_match = re.search(r"\bin\s+the?\s*([a-z0-9][a-z0-9 _/-]*)\s+region\b", q_text)
    if region_match:
        value = normalize_text(region_match.group(1))
        key = ("region", "=", value)
        if value and key not in seen:
            seen.add(key)
            candidates.append(
                FilterCandidate(
                    raw_value=value,
                    operator="=",
                    column_hint="region",
                    kind="equality",
                    confidence=0.88,
                    source="region_phrase",
                    reasons=["Matched 'in ... region' phrase."],
                )
            )

    generic_for_match = re.search(r"\bfor\s+([a-z0-9][a-z0-9 _/-]*)\b(?=\s+(?:in|where|with|and|$))", q_text)
    if generic_for_match:
        value = normalize_text(generic_for_match.group(1))
        singular_value = value[:-1] if value.endswith("s") else value
        if value and value not in reserved_values and singular_value not in reserved_values and ("for", "=", value) not in seen:
            candidates.append(
                FilterCandidate(
                    raw_value=value,
                    operator="=",
                    column_hint=None,
                    kind="equality",
                    confidence=0.52,
                    source="generic_for_phrase",
                    reasons=["Matched generic 'for <value>' phrase."],
                )
            )

    comparison_patterns = [
        r"\b([a-z0-9][a-z0-9 _/-]*)\s+vs\.?\s+([a-z0-9][a-z0-9 _/-]*)\b",
        r"\bbetween\s+([a-z0-9][a-z0-9 _/-]*)\s+and\s+([a-z0-9][a-z0-9 _/-]*)\b",
    ]
    for pattern in comparison_patterns:
        for match in re.finditer(pattern, q_text):
            for raw_piece in match.groups():
                value = _clean_comparison_piece(raw_piece)
                singular_value = value[:-1] if value.endswith("s") else value
                if not value or value in reserved_values or singular_value in reserved_values:
                    continue
                key = ("comparison", "=", value)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    FilterCandidate(
                        raw_value=value,
                        operator="=",
                        column_hint=None,
                        kind="equality",
                        confidence=0.64,
                        source="comparison_phrase",
                        reasons=["Matched comparison phrase value."],
                    )
                )

    mentions_invoice_status = bool(re.search(r"\binvoice\s+status\b", q_text))
    status_column = _pick_status_like_column(context, target_sheet)
    if status_column:
        billing_status_patterns = [
            (r"\bunder[ -]?billed\b", "under billed"),
            (r"\bover[ -]?billed\b", "over billed"),
            (r"\bunbilled\b|\bunpaid\b|\bnot billed\b|\bwithout invoice\b", "unbilled"),
            (r"\bbilling delay\b|\blagged\b|\bdelayed\b|\boverdue\b", "billing delay"),
        ]
        if not mentions_invoice_status:
            for pattern, value in billing_status_patterns:
                key = (normalize_name(status_column), "=", value)
                if re.search(pattern, q_text) and key not in seen:
                    seen.add(key)
                    candidates.append(
                        FilterCandidate(
                            raw_value=value,
                            operator="=",
                            column_hint=status_column,
                            kind="equality",
                            confidence=0.9,
                            source="billing_status_phrase",
                            reasons=[f"Matched billing status phrase for column '{status_column}'."],
                        )
                    )

    invoice_status_column = _pick_invoice_status_like_column(context, target_sheet)
    if invoice_status_column:
        invoice_status_patterns = [
            (r"\binvoice\s+status\s+under[ -]?billed\b", "under billed"),
            (r"\binvoice\s+status\s+over[ -]?billed\b", "over billed"),
            (r"\binvoice\s+status\s+(?:unbilled|unpaid|not billed|without invoice)\b", "unbilled"),
            (r"\binvoice\s+status\s+(?:billing delay|lagged|delayed|overdue)\b", "billing delay"),
        ]
        for pattern, value in invoice_status_patterns:
            key = (normalize_name(invoice_status_column), "contains", value)
            if re.search(pattern, q_text) and key not in seen:
                seen.add(key)
                candidates.append(
                    FilterCandidate(
                        raw_value=value,
                        operator="contains",
                        column_hint=invoice_status_column,
                        kind="equality",
                        confidence=0.94,
                        source="invoice_status_phrase",
                        reasons=[f"Matched invoice status phrase for column '{invoice_status_column}'."],
                    )
                )

    return candidates