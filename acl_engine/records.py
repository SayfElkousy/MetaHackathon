"""Parameter records (v4 Section 1.1). Bare numbers wrap to user_design/unknown."""
import math

ORIGINS = {"measured", "literature_example", "user_design",
           "prototype_placeholder", "derived", "unknown"}
CONFIDENCES = {"high", "medium", "low", "unknown"}


def _blank(value, unit, origin):
    return {"value": value, "unit": unit, "origin": origin,
            "source": None, "method": None, "reference_position": None,
            "confidence": "unknown",
            "uncertainty": {"type": "none", "value": None}}


def as_record(x, unit, field, default_origin="user_design"):
    if isinstance(x, dict) and "value" in x:
        v = x.get("value", None)
        if v is not None and not isinstance(v, (int, float, str, bool)):
            raise ValueError("%s record value has bad type %r" % (field, v))
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("%s record value must be finite" % field)
        origin = x.get("origin", default_origin)
        if origin not in ORIGINS:
            raise ValueError("%s bad origin %r" % (field, origin))
        conf = x.get("confidence", "unknown")
        if conf not in CONFIDENCES:
            raise ValueError("%s bad confidence %r" % (field, conf))
        return {"value": v, "unit": x.get("unit", unit),
                "origin": origin, "source": x.get("source", None),
                "method": x.get("method", None),
                "reference_position": x.get("reference_position", None),
                "confidence": conf,
                "uncertainty": x.get("uncertainty", {"type": "none", "value": None})}
    if x is None:
        return _blank(None, unit, "unknown")
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("%s must be finite" % field)
        return _blank(float(x), unit, default_origin)
    if isinstance(x, (str, bool)):
        return _blank(x, unit, default_origin)
    if isinstance(x, list):
        return _blank(list(x), unit, default_origin)
    raise ValueError("%s has unsupported type %r" % (field, type(x)))


def get_value(rec):
    if isinstance(rec, dict) and "value" in rec:
        return rec["value"]
    return rec


def is_unknown(rec):
    if isinstance(rec, dict) and "value" in rec:
        return rec["value"] is None
    return rec is None
