from datetime import datetime
from typing import Dict
import uuid


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _new_correlation_id() -> str:
    return str(uuid.uuid4())


def make_error(message: str) -> Dict[str, str]:
    return {
        "type": "error",
        "message": message,
        "timestamp": _now_iso(),
        "correlation_id": _new_correlation_id(),
    }


def make_data_error(data_type: str, message: str) -> Dict[str, str]:
    return {
        "type": "data_error",
        "data_type": data_type,
        "error": message,
        "timestamp": _now_iso(),
        "correlation_id": _new_correlation_id(),
    }
