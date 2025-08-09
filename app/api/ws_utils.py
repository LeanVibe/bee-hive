from datetime import datetime
from typing import Dict


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def make_error(message: str) -> Dict[str, str]:
    return {
        "type": "error",
        "message": message,
        "timestamp": _now_iso(),
    }


def make_data_error(data_type: str, message: str) -> Dict[str, str]:
    return {
        "type": "data_error",
        "data_type": data_type,
        "error": message,
        "timestamp": _now_iso(),
    }
