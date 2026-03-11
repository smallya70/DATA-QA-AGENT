from typing import Optional

from pydantic import BaseModel, Field, field_validator


class QuerySpec(BaseModel):
    intent: str
    query_type: str | None = None
    pattern_id: str | None = None
    target_sheet: str | None = None
    columns: list[str] = Field(default_factory=list)
    measures: list[dict] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    filters: list[dict] = Field(default_factory=list)
    order_by: list[dict] = Field(default_factory=list)
    limit: int | None = None
    time_grain: str | None = None
    join_path: list[dict] = Field(default_factory=list)
    partition_by: list[str] = Field(default_factory=list)
    rank_dimension: str | None = None
    rank_limit: int | None = None
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)
    source: str | None = None

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("intent is required")
        return text

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if int(value) <= 0:
            raise ValueError("limit must be greater than 0")
        return int(value)

    @field_validator("rank_limit")
    @classmethod
    def validate_rank_limit(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if int(value) <= 0:
            raise ValueError("rank_limit must be greater than 0")
        return int(value)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        number = float(value)
        if number < 0.0 or number > 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return number
