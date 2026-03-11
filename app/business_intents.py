from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from app.planning_context import normalize_name


SUM_INTENT = ["total", "sum", "overall", "combined", "aggregate"]
AVG_INTENT = ["average", "avg", "mean"]
MAX_INTENT = ["highest", "maximum", "max", "largest", "top", "peak"]
MIN_INTENT = ["lowest", "minimum", "min", "smallest", "least"]
COUNT_INTENT = ["count", "number of", "how many"]
DISTINCT_INTENT = ["distinct", "unique"]
BEST_RANKING_INTENT = ["best", "best performing"]
WORST_RANKING_INTENT = ["worst", "worst performing"]
TOP_N_MAX_OVERRIDE_INTENT = ["highest", "maximum", "max", "largest single", "peak"]
BOTTOM_N_MIN_OVERRIDE_INTENT = ["lowest", "minimum", "min", "smallest", "least", "lowest single", "trough"]


BUSINESS_SYNONYMS: dict[str, dict[str, list[str]]] = {
    "metrics": {
        "amount": ["amount", "value", "revenue", "sales", "turnover", "gross sales", "order value", "inventory value", "value gap", "variance", "total", "price", "spend", "cost"],
        "cost": ["cost", "costs", "spend", "expense", "expenses"],
        "profit": ["profit", "margin", "earnings"],
        "quantity": ["quantity", "qty", "units", "unit", "volume"],
        "count": ["count", "number", "how many", "rows", "records"],
        "discount": ["discount", "rebate"],
        "tax": ["tax", "vat", "gst"],
    },
    "dimensions": {
        "status": ["status", "state", "stage"],
        "invoice": ["invoice", "invoices", "invoice id", "invoice number", "bill", "bill no"],
        "country": ["country", "countries", "market", "geography", "geo"],
        "region": ["region", "regions", "territory", "location", "state", "area"],
        "product": ["product", "products", "item", "items", "sku", "material", "service"],
        "customer": ["customer", "customers", "client", "clients", "account", "accounts", "buyer", "buyers"],
        "date": ["date", "time", "period"],
        "year": ["year", "annual", "yearly"],
        "month": ["month", "monthly"],
        "quarter": ["quarter", "quarterly"],
    },
    "roles": {
        "revenue": ["amount", "value", "revenue", "sales", "turnover", "gross sales", "order value", "total", "price"],
        "cost": ["cost", "costs", "spend", "expense", "expenses"],
        "profit": ["profit", "margin", "earnings"],
        "value_gap": ["value gap", "variance", "gap"],
        "inventory_value": ["inventory value", "stock value"],
        "quantity": ["quantity", "qty", "units", "volume"],
        "status": ["status", "state", "stage"],
        "invoice_id": ["invoice", "invoices", "invoice id", "invoice number", "bill", "bill no"],
        "country": ["country", "market", "geography"],
        "region": ["region", "territory", "location", "state", "area"],
        "product": ["product", "item", "sku", "material", "service"],
        "customer_id": ["customer", "client", "account", "buyer"],
        "date": ["date", "time", "month", "quarter", "year", "period"],
        "service_date": ["service date", "service time"],
    },
}


INTENT_PATTERNS: dict[str, dict[str, Any]] = {
    "row_preview": {"query_shape": "row_preview", "answer_type": "table", "aggregation": None},
    "describe_sheet": {"query_shape": "describe_sheet", "answer_type": "sheet_profile", "aggregation": None},
    "projection": {"query_shape": "projection", "answer_type": "table", "aggregation": None},
    "list_rows": {"query_shape": "list_rows", "answer_type": "row_list", "aggregation": None},
    "total_x": {"query_shape": "aggregate_scalar", "answer_type": "scalar_value", "aggregation": "sum"},
    "total_x_for_y": {"query_shape": "aggregate_scalar", "answer_type": "filtered_scalar", "aggregation": "sum"},
    "x_by_y": {"query_shape": "aggregate_grouped", "answer_type": "grouped_table", "aggregation": "sum"},
    "max_by_y": {"query_shape": "aggregate_grouped", "answer_type": "grouped_table", "aggregation": "max"},
    "min_by_y": {"query_shape": "aggregate_grouped", "answer_type": "grouped_table", "aggregation": "min"},
    "top_n_x_by_y": {"query_shape": "aggregate_grouped", "answer_type": "ranked_table", "aggregation": "sum"},
    "highest_x": {"query_shape": "aggregate_grouped", "answer_type": "ranked_table", "aggregation": "max"},
    "lowest_x": {"query_shape": "aggregate_grouped", "answer_type": "ranked_table", "aggregation": "min"},
    "count_of_x": {"query_shape": "aggregate_scalar", "answer_type": "scalar_count", "aggregation": "count"},
    "count_of_x_where_y": {"query_shape": "aggregate_scalar", "answer_type": "filtered_count", "aggregation": "count"},
    "count_distinct_x": {"query_shape": "aggregate_scalar", "answer_type": "scalar_count", "aggregation": "count_distinct"},
    "compare_x_across_y": {"query_shape": "comparison", "answer_type": "comparison_table", "aggregation": "sum"},
    "trend_of_x_over_time": {"query_shape": "trend_over_time", "answer_type": "trend_series", "aggregation": "sum"},
    "average_x_by_y": {"query_shape": "aggregate_grouped", "answer_type": "grouped_table", "aggregation": "avg"},
    "unsupported": {"query_shape": "unknown", "answer_type": "guidance", "aggregation": None},
}


@dataclass(slots=True)
class CanonicalIntent:
    name: str
    query_shape: str
    aggregation: str | None
    answer_type: str
    confidence: float
    ranking: str | None = None
    requested_top_n: int | None = None
    time_grain: str | None = None
    metric_families: list[str] = field(default_factory=list)
    dimension_families: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _matches_intent_phrase(question_text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower())
    pattern = escaped.replace(r"\ ", r"\s+")
    return bool(re.search(rf"\b{pattern}\b", question_text))


def matched_intent_keywords(question: str) -> dict[str, list[str]]:
    q_text = str(question or "").lower()
    keyword_map = {
        "sum": SUM_INTENT,
        "avg": AVG_INTENT,
        "max": MAX_INTENT,
        "min": MIN_INTENT,
        "count": COUNT_INTENT,
        "distinct": DISTINCT_INTENT,
        "best_rank": BEST_RANKING_INTENT,
        "worst_rank": WORST_RANKING_INTENT,
    }
    return {
        family: [phrase for phrase in phrases if _matches_intent_phrase(q_text, phrase)]
        for family, phrases in keyword_map.items()
    }


def has_distinct_intent(question: str) -> bool:
    return bool(matched_intent_keywords(question).get("distinct"))


def top_n_aggregation_for_question(question: str) -> str:
    q_text = str(question or "").lower()
    if any(_matches_intent_phrase(q_text, phrase) for phrase in TOP_N_MAX_OVERRIDE_INTENT):
        return "max"
    return "sum"


def bottom_n_aggregation_for_question(question: str) -> str:
    q_text = str(question or "").lower()
    if any(_matches_intent_phrase(q_text, phrase) for phrase in BOTTOM_N_MIN_OVERRIDE_INTENT):
        return "min"
    return "sum"


def best_or_worst_ranking_direction(question: str) -> str | None:
    q_text = str(question or "").lower()
    if any(_matches_intent_phrase(q_text, phrase) for phrase in BEST_RANKING_INTENT):
        return "desc"
    if any(_matches_intent_phrase(q_text, phrase) for phrase in WORST_RANKING_INTENT):
        return "asc"
    return None


def best_or_worst_ranking_aggregation_for_question(question: str, direction: str) -> str:
    return top_n_aggregation_for_question(question) if direction == "desc" else bottom_n_aggregation_for_question(question)


def aggregation_for_question(question: str) -> str | None:
    q_text = str(question or "").lower()
    if any(_matches_intent_phrase(q_text, phrase) for phrase in COUNT_INTENT):
        return "count"
    if any(_matches_intent_phrase(q_text, phrase) for phrase in AVG_INTENT):
        return "avg"
    if any(_matches_intent_phrase(q_text, phrase) for phrase in MAX_INTENT):
        return "max"
    if any(_matches_intent_phrase(q_text, phrase) for phrase in MIN_INTENT):
        return "min"
    if any(_matches_intent_phrase(q_text, phrase) for phrase in SUM_INTENT):
        return "sum"
    return None


def has_explicit_aggregation_language(question: str) -> bool:
    return aggregation_for_question(question) is not None


def has_group_by_intent(question: str) -> bool:
    q_text = str(question or "").lower()
    return bool(re.search(r"\b(by|per|for each|group by|breakdown|split by|across|wise)\b", q_text))


def has_row_preview_intent(question: str) -> bool:
    q_text = str(question or "").strip().lower()
    if not q_text:
        return False
    if re.match(r"^\s*preview\b", q_text):
        return True
    preview_verb = bool(re.search(r"\b(show|display|list|return)\b", q_text))
    row_noun = bool(re.search(r"\b(rows?|records?)\b", q_text))
    explicit_row_limit = bool(re.search(r"\b(first|top)\s+\d+\s*(rows?|records?)?\b", q_text))
    return preview_verb and (row_noun or explicit_row_limit)


def has_describe_sheet_intent(question: str) -> bool:
    q_text = str(question or "").strip().lower()
    if not q_text:
        return False
    if not re.match(r"^\s*(describe|summarize|summary of|tell me about)\b", q_text):
        return False
    if has_group_by_intent(q_text) or has_explicit_aggregation_language(q_text):
        return False
    if re.search(r"\b(highest|top|maximum|max|lowest|minimum|min|count\s+of|how\s+many|number\s+of)\b", q_text):
        return False
    return True


def _question_tokens(question: str) -> set[str]:
    return {token for token in normalize_name(question).split("_") if token}


def _contains_alias(question: str, aliases: list[str]) -> bool:
    q_norm = normalize_name(question)
    q_tokens = _question_tokens(question)
    for alias in aliases:
        normalized_alias = normalize_name(alias)
        parts = [part for part in normalized_alias.split("_") if part]
        if not parts:
            continue
        if len(parts) == 1:
            if parts[0] in q_tokens:
                return True
        elif normalized_alias in q_norm:
            return True
    return False


def metric_families_for_question(question: str) -> list[str]:
    families: list[str] = []
    for family, aliases in BUSINESS_SYNONYMS["metrics"].items():
        if _contains_alias(question, aliases):
            families.append(family)
    if "count" in families:
        return ["count"]
    if "amount" in families and len(families) > 1:
        families = [family for family in families if family != "amount"]
    return families


def dimension_families_for_question(question: str) -> list[str]:
    families: list[str] = []
    for family, aliases in BUSINESS_SYNONYMS["dimensions"].items():
        if _contains_alias(question, aliases):
            families.append(family)
    return families


def semantic_aliases(role_name: str) -> list[str]:
    normalized_role = normalize_name(role_name)
    aliases = list(BUSINESS_SYNONYMS["roles"].get(normalized_role, []))
    if normalized_role in BUSINESS_SYNONYMS["dimensions"]:
        aliases.extend(BUSINESS_SYNONYMS["dimensions"][normalized_role])
    if normalized_role in BUSINESS_SYNONYMS["metrics"]:
        aliases.extend(BUSINESS_SYNONYMS["metrics"][normalized_role])
    aliases.append(normalized_role.replace("_", " "))
    return list(dict.fromkeys(aliases))


def role_matches_question(role_name: str, question: str) -> bool:
    return _contains_alias(question, semantic_aliases(role_name))


def family_matches_column_or_role(family: str, column_name: str, role_name: str | None = None) -> bool:
    aliases = list(BUSINESS_SYNONYMS["metrics"].get(family, [])) + list(BUSINESS_SYNONYMS["dimensions"].get(family, []))
    column_norm = normalize_name(column_name)
    if any(normalize_name(alias) in column_norm or column_norm in normalize_name(alias) for alias in aliases):
        return True
    if role_name:
        role_norm = normalize_name(role_name)
        if family == role_norm:
            return True
        if any(normalize_name(alias) in role_norm for alias in aliases):
            return True
    return False


def detect_canonical_intent(question: str) -> CanonicalIntent:
    q_text = str(question or "").strip().lower()
    metric_families = metric_families_for_question(question)
    dimension_families = dimension_families_for_question(question)
    reasons: list[str] = []

    row_preview_intent = has_row_preview_intent(q_text)
    describe_sheet_intent = has_describe_sheet_intent(q_text)

    count_intent = bool(re.search(r"\b(count\s+of|count\s+where|count\s+by|how\s+many|number\s+of)\b", q_text))
    avg_intent = bool(re.search(r"\b(avg|average|mean)\b", q_text))
    comparison_intent = bool(re.search(r"\b(compare|comparison|vs\.?|versus|across)\b", q_text))
    grouping_intent = has_group_by_intent(q_text)
    ranking_high = bool(re.search(r"\b(highest|top|maximum|max|most)\b", q_text))
    ranking_low = bool(re.search(r"\b(lowest|minimum|min|least|bottom)\b", q_text))
    ranking_best = any(_matches_intent_phrase(q_text, phrase) for phrase in BEST_RANKING_INTENT)
    ranking_worst = any(_matches_intent_phrase(q_text, phrase) for phrase in WORST_RANKING_INTENT)
    filter_intent = bool(re.search(r"\b(for|where|in|with)\b", q_text))
    time_intent = bool(re.search(r"\b(trend|over time|by month|by year|by quarter|monthly|yearly|quarterly)\b", q_text))
    list_intent = bool(re.search(r"\b(list|show|display|give|give me|return)\b", q_text))
    grouped_list_dimension_words = bool(re.search(r"\b(products|customers|regions|countries|markets|locations|items|skus|services|stages|statuses)\b", q_text))
    aggregate_intent = aggregation_for_question(q_text)
    per_group_extrema = bool(re.search(r"\b(for each|each|wise)\b", q_text))

    requested_top_n = None
    top_n_match = re.search(r"\btop\s+(\d+)\b", q_text)
    if top_n_match:
        requested_top_n = max(1, int(top_n_match.group(1)))

    if row_preview_intent:
        reasons.append("Detected explicit row preview request.")
        return CanonicalIntent("row_preview", "row_preview", None, "table", 0.98, metric_families=[], dimension_families=dimension_families, reasons=reasons)

    if describe_sheet_intent:
        reasons.append("Detected describe-sheet request.")
        return CanonicalIntent("describe_sheet", "describe_sheet", None, "sheet_profile", 0.97, metric_families=[], dimension_families=dimension_families, reasons=reasons)

    time_grain = None
    if "month" in q_text:
        time_grain = "month"
    elif "quarter" in q_text:
        time_grain = "quarter"
    elif "year" in q_text or time_intent:
        time_grain = "year"

    if comparison_intent and not count_intent:
        reasons.append("Detected comparison language for a business metric question.")
        return CanonicalIntent("compare_x_across_y", "comparison", "sum", "comparison_table", 0.93, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if time_intent:
        reasons.append("Detected time-series language.")
        return CanonicalIntent("trend_of_x_over_time", "trend_over_time", "count" if count_intent else ("avg" if avg_intent else "sum"), "trend_series", 0.94, time_grain=time_grain, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if avg_intent and grouping_intent:
        reasons.append("Detected average metric with grouping language.")
        return CanonicalIntent("average_x_by_y", "aggregate_grouped", "avg", "grouped_table", 0.92, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if ranking_high and grouping_intent and per_group_extrema:
        reasons.append("Detected grouped maximum question.")
        return CanonicalIntent("max_by_y", "aggregate_grouped", "max", "grouped_table", 0.94, ranking="desc", metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if ranking_low and grouping_intent and per_group_extrema:
        reasons.append("Detected grouped minimum question.")
        return CanonicalIntent("min_by_y", "aggregate_grouped", "min", "grouped_table", 0.94, ranking="asc", metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if requested_top_n and grouping_intent:
        reasons.append("Detected explicit top-N grouped question.")
        return CanonicalIntent("top_n_x_by_y", "aggregate_grouped", top_n_aggregation_for_question(q_text), "ranked_table", 0.93, ranking="desc", requested_top_n=requested_top_n, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if ranking_best and grouping_intent:
        reasons.append("Detected best-ranking grouped question.")
        return CanonicalIntent("top_n_x_by_y", "aggregate_grouped", "sum", "ranked_table", 0.91, ranking="desc", requested_top_n=1, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if ranking_worst and grouping_intent:
        reasons.append("Detected worst-ranking grouped question.")
        return CanonicalIntent("top_n_x_by_y", "aggregate_grouped", "sum", "ranked_table", 0.91, ranking="asc", requested_top_n=1, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if ranking_high:
        reasons.append("Detected highest/top style ranking question.")
        return CanonicalIntent("highest_x", "aggregate_grouped", "max", "ranked_table", 0.91, ranking="desc", requested_top_n=1, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if ranking_low:
        reasons.append("Detected lowest/bottom style ranking question.")
        return CanonicalIntent("lowest_x", "aggregate_grouped", "min", "ranked_table", 0.91, ranking="asc", requested_top_n=1, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if count_intent and filter_intent:
        reasons.append("Detected count question with filter language.")
        return CanonicalIntent("count_of_x_where_y", "aggregate_scalar", "count", "filtered_count", 0.9, metric_families=["count"], dimension_families=dimension_families, reasons=reasons)
    if count_intent:
        reasons.append("Detected count question.")
        return CanonicalIntent("count_of_x", "aggregate_scalar", "count", "scalar_count", 0.9, metric_families=["count"], dimension_families=dimension_families, reasons=reasons)
    explicit_dimension_words = bool(re.search(r"\b(product|products|region|regions|country|countries|customer|customers|client|clients|status|stage|state|invoice|invoices|bill|bills)\b", q_text))
    if list_intent and metric_families and grouped_list_dimension_words:
        reasons.append("Detected list-style grouped business question.")
        return CanonicalIntent("x_by_y", "aggregate_grouped", "sum", "grouped_table", 0.87, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if list_intent and not metric_families and (dimension_families or explicit_dimension_words):
        reasons.append("Detected row-list question without explicit metric intent.")
        return CanonicalIntent("list_rows", "list_rows", None, "row_list", 0.86, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if grouping_intent:
        reasons.append("Detected grouped aggregate question.")
        return CanonicalIntent("x_by_y", "aggregate_grouped", aggregate_intent or "sum", "grouped_table", 0.88, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if filter_intent:
        reasons.append("Detected filtered scalar aggregate question.")
        return CanonicalIntent("total_x_for_y", "aggregate_scalar", "sum", "filtered_scalar", 0.84, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    if metric_families:
        reasons.append("Detected scalar business metric question.")
        return CanonicalIntent("total_x", "aggregate_scalar", "sum", "scalar_value", 0.82, metric_families=metric_families, dimension_families=dimension_families, reasons=reasons)
    return CanonicalIntent("unsupported", "unknown", None, "guidance", 0.3, metric_families=metric_families, dimension_families=dimension_families, reasons=["No supported deterministic business intent matched."])