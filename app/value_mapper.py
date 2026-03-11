from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any
import re

from app.business_intents import dimension_families_for_question, family_matches_column_or_role
from app.filter_extractor import FilterCandidate
from app.planning_context import PlanningContext, normalize_name, normalize_text


@dataclass(slots=True)
class ValueMapping:
    sheet: str
    column: str
    value: Any
    operator: str = "="
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    ambiguous_columns: list[str] = field(default_factory=list)
    source: str = "deterministic"
    kind: str = "equality"


def _column_hint_score(column_name: str, candidate: FilterCandidate) -> float:
    if not candidate.column_hint:
        return 0.0
    hint = normalize_name(candidate.column_hint)
    column_norm = normalize_name(column_name)
    if hint == column_norm:
        return 0.25
    if hint in column_norm or column_norm in hint:
        return 0.12
    return 0.0


def _semantic_role_score(sheet_name: str, column_name: str, context: PlanningContext) -> float:
    role_map = context.semantic_roles.get(sheet_name) or {}
    if any(str(mapped_column) == str(column_name) for mapped_column in role_map.values()):
        return 0.08
    return 0.0


def _role_name_for_column(sheet_name: str, column_name: str, context: PlanningContext) -> str | None:
    role_map = context.semantic_roles.get(sheet_name) or {}
    for role_name, mapped_column in role_map.items():
        if str(mapped_column) == str(column_name):
            return str(role_name)
    return None


def _question_family_score(question: str, sheet_name: str, column_name: str, context: PlanningContext) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    role_name = _role_name_for_column(sheet_name, column_name, context)
    for family in dimension_families_for_question(question):
        if family_matches_column_or_role(family, column_name, role_name):
            score += 0.22
            reasons.append(f"Question language aligns '{column_name}' to business family '{family}'.")
    return score, reasons


def _sample_value_score(raw_value: str, sample_value: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if raw_value == sample_value:
        reasons.append("Exact categorical sample match.")
        return 0.72, reasons
    similarity = SequenceMatcher(None, raw_value, sample_value).ratio()
    if similarity >= 0.92:
        reasons.append(f"High normalized similarity to sample value '{sample_value}'.")
        return 0.48, reasons
    if similarity >= 0.80:
        reasons.append(f"Moderate normalized similarity to sample value '{sample_value}'.")
        return 0.28, reasons
    return 0.0, reasons


def _scan_question_for_sample_values(
    question: str,
    context: PlanningContext,
    target_sheet: str | None,
    used_values: set[str],
) -> list[FilterCandidate]:
    question_text = normalize_text(question)
    if not question_text:
        return []

    sheet_names = [target_sheet] if target_sheet else list(context.sampled_categorical_values_by_sheet.keys())
    candidates: list[FilterCandidate] = []
    for sheet_name in [sheet for sheet in sheet_names if sheet]:
        sampled_values = context.sampled_categorical_values_by_sheet.get(sheet_name) or {}
        for column_name, values in sampled_values.items():
            for value in values:
                normalized_value = normalize_text(value)
                if not normalized_value or normalized_value in used_values:
                    continue
                if re.search(rf"\b{re.escape(normalized_value)}\b", question_text):
                    candidates.append(
                        FilterCandidate(
                            raw_value=normalized_value,
                            operator="=",
                            column_hint=column_name,
                            kind="equality",
                            confidence=0.66,
                            source="sample_value_match",
                            reasons=[f"Question contains sampled categorical value '{normalized_value}'."],
                        )
                    )
                    used_values.add(normalized_value)
    return candidates


def map_values_to_columns(
    question: str,
    context: PlanningContext,
    *,
    target_sheet: str | None,
    candidates: list[FilterCandidate],
) -> tuple[list[ValueMapping], list[str]]:
    warnings: list[str] = []
    used_values = {normalize_text(candidate.raw_value) for candidate in candidates if normalize_text(candidate.raw_value)}
    all_candidates = list(candidates) + _scan_question_for_sample_values(question, context, target_sheet, used_values)

    sheet_names = [target_sheet] if target_sheet else list(context.sheets)
    mappings: list[ValueMapping] = []
    for candidate in all_candidates:
        raw_value = normalize_text(candidate.raw_value)
        if not raw_value:
            continue

        scored_matches: list[tuple[float, str, str, list[str]]] = []
        for sheet_name in [sheet for sheet in sheet_names if sheet]:
            categorical_columns = context.categorical_columns_by_sheet.get(sheet_name) or []
            sampled_values = context.sampled_categorical_values_by_sheet.get(sheet_name) or {}
            for column_name in categorical_columns:
                sample_values = sampled_values.get(column_name) or []
                best_score = 0.0
                best_reasons: list[str] = []
                for sample_value in sample_values:
                    sample_score, sample_reasons = _sample_value_score(raw_value, normalize_text(sample_value))
                    if sample_score > best_score:
                        best_score = sample_score
                        best_reasons = list(sample_reasons)
                best_score += _column_hint_score(column_name, candidate)
                if candidate.column_hint and _column_hint_score(column_name, candidate) > 0:
                    best_reasons.append(f"Column hint favored '{column_name}'.")
                semantic_score = _semantic_role_score(sheet_name, column_name, context)
                best_score += semantic_score
                if semantic_score:
                    best_reasons.append(f"Semantic roles include '{column_name}'.")
                family_score, family_reasons = _question_family_score(question, sheet_name, column_name, context)
                best_score += family_score
                best_reasons.extend(family_reasons)
                if best_score > 0:
                    scored_matches.append((best_score, sheet_name, column_name, best_reasons))

        if not scored_matches:
            continue

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        best_score, best_sheet, best_column, best_reasons = scored_matches[0]
        ambiguous_columns: list[str] = []
        if len(scored_matches) > 1 and abs(scored_matches[0][0] - scored_matches[1][0]) <= 0.08:
            ambiguous_columns = [f"{sheet}.{column}" for _, sheet, column, _ in scored_matches[:3]]
            warnings.append(f"Value '{raw_value}' could map to multiple columns: {', '.join(ambiguous_columns)}")
            best_score -= 0.12

        mappings.append(
            ValueMapping(
                sheet=best_sheet,
                column=best_column,
                value=raw_value,
                operator=candidate.operator,
                confidence=max(0.0, min(1.0, best_score)),
                reasons=list(dict.fromkeys(candidate.reasons + best_reasons)),
                ambiguous_columns=ambiguous_columns,
                source=candidate.source,
                kind=candidate.kind,
            )
        )

    deduped: list[ValueMapping] = []
    seen: set[tuple[str, str, str, str]] = set()
    for mapping in mappings:
        key = (mapping.sheet, mapping.column, str(mapping.operator), normalize_text(mapping.value))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(mapping)
    return deduped, warnings