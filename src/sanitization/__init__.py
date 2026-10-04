"""Policy-controlled removal of restricted source fields."""

from .policy import sanitize_record

__all__ = ["sanitize_record"]
