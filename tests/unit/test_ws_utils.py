import re
from app.api.ws_utils import make_error, make_data_error


def test_make_error_includes_timestamp_iso_and_message():
    err = make_error("oops")
    assert err["type"] == "error"
    assert err["message"] == "oops"
    assert isinstance(err.get("timestamp"), str)
    # basic ISO-8601 pattern check (not strict)
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*", err["timestamp"]) is not None


def test_make_data_error_shape():
    derr = make_data_error("unknown", "bad data")
    assert derr["type"] == "data_error"
    assert derr["data_type"] == "unknown"
    assert derr["error"] == "bad data"
    assert isinstance(derr.get("timestamp"), str)
