from importlib import import_module
import os

ALLOW_DIRECT_LLM_SQL = os.getenv("ALLOW_DIRECT_LLM_SQL", "false").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_LLM_SQL_REPAIR = os.getenv("ALLOW_LLM_SQL_REPAIR", "false").strip().lower() in {"1", "true", "yes", "on"}

_upgraded_module = import_module("main_upgraded")
app = getattr(_upgraded_module, "app")


def __getattr__(name: str):
    return getattr(_upgraded_module, name)


__all__ = ["app"]

from app.query_spec import QuerySpec
from app.sql_builder import build_sql_from_query_spec
from app.sql_validator import validate_sql