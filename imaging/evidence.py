"""Evidence -> scaffold-parameter mapping layer.

Sources and how certain they are (never treated as equal):
    measured   reviewed, calibrated MRI measurement (annotation with status 'reviewed', millimetres)
    reported   finding extracted from a radiology report the user pasted (the report is data, not instructions)
    inferred   model output (e.g. KneeScan tear status); shown, never applied without the user's confirmation
    assumption a labelled design assumption (e.g. 'proximal' -> tear position 0.8)

Each mapping states its required source, target parameter, transform, assumptions, constraints (the engine's own
parameter limits via validate_params) and whether review is required. Values the evidence cannot determine keep their
current setting. Microscopic features are explicitly not mapped (see NOT_MAPPED).
"""
import copy
import re

from acl_engine.v2 import design as D

CERTAINTY_RANK = {"measured": 3, "reported": 2, "inferred": 1, "assumption": 0}

# measurement kinds the annotation tools offer, with what they mean physically
MEASUREMENT_KINDS = {
    "acl_length": "ACL length: straight-line distance between the femoral and tibial attachment centres",
    "acl_width": "ACL mid-substance width (sagittal plane, perpendicular to the ligament)",
    "tear_gap": "Gap between the torn stumps (shortest distance)",
    "tib_fp_length": "Tibial footprint anterior-posterior length",
    "tib_fp_width": "Tibial footprint medial-lateral width",
    "fem_fp_length": "Femoral footprint length (along the notch wall)",
    "fem_fp_width": "Femoral footprint width",
    "other": "Other distance (not mapped)",
}
LANDMARK_KINDS = {
    "femoral_attachment": "Femoral ACL attachment centre",
    "tibial_attachment": "Tibial ACL attachment centre",
    "femoral_stump_end": "Femoral stump end",
    "tibial_stump_end": "Tibial stump end",
    "other": "Other landmark",
}

LOCATION_S = {"proximal": 0.8, "midsubstance": 0.5, "distal": 0.2}

MAPPINGS = [
    {"id": "acl_length", "needs": "acl_length", "targets": {"lig.length_target_mm": "value", "lig.length_mode": "target"},
     "transform": "lig.length_target_mm = measured distance (mm); lig.length_mode = target",
     "relation": "assumption",
     "assumptions": ["The MRI distance is a straight line between attachment centres; the engine's target is the centreline "
                     "length, which it meets by solving the anterior bow. A bowed ligament is slightly longer than the chord."],
     "review_required": False, "levels": [0]},
    {"id": "acl_width", "needs": "acl_width", "targets": {"lig.width_profile": "width_profile"},
     "transform": "template width profile scaled so its mid-substance median equals the measured width",
     "relation": "assumption",
     "assumptions": ["Width measured in one MRI plane is taken as the ligament width; the profile shape along the ligament "
                     "stays the template's."],
     "requires": {"lig.section_mode": "width_thickness"}, "review_required": False, "levels": [0]},
    {"id": "tear_gap", "needs": "tear_gap", "targets": {"injury.gap_mm": "value"}, "transform": "direct (mm)",
     "relation": "direct", "assumptions": [], "review_required": False, "levels": [0]},
    {"id": "tib_fp_length", "needs": "tib_fp_length", "targets": {"fp.tib.length_mm": "value"}, "transform": "direct (mm)",
     "relation": "direct", "assumptions": [], "requires_not": {"fp.tib.shape": "polygon"}, "review_required": False, "levels": [0]},
    {"id": "tib_fp_width", "needs": "tib_fp_width", "targets": {"fp.tib.width_mm": "value"}, "transform": "direct (mm)",
     "relation": "direct", "assumptions": [], "requires_not": {"fp.tib.shape": "polygon"}, "review_required": False, "levels": [0]},
    {"id": "fem_fp_length", "needs": "fem_fp_length", "targets": {"fp.fem.length_mm": "value"}, "transform": "direct (mm)",
     "relation": "direct", "assumptions": [], "requires_not": {"fp.fem.shape": "polygon"}, "review_required": False, "levels": [0]},
    {"id": "fem_fp_width", "needs": "fem_fp_width", "targets": {"fp.fem.width_mm": "value"}, "transform": "direct (mm)",
     "relation": "direct", "assumptions": [], "requires_not": {"fp.fem.shape": "polygon"}, "review_required": False, "levels": [0]},
    {"id": "tear_type", "needs": "tear_type", "targets": {"injury.type": "value"}, "transform": "category (intact / partial / complete)",
     "relation": "direct", "assumptions": [], "review_required": False, "levels": [0]},
    {"id": "tear_bundle", "needs": "tear_bundle", "targets": {"injury.involvement": "value"},
     "transform": "AM / PL bundle involvement (partial tears only)", "relation": "direct", "assumptions": [],
     "requires_value": {"injury.type": "partial"}, "review_required": False, "levels": [0]},
    {"id": "tear_location", "needs": "tear_location", "targets": {"injury.center_s": "location_s"},
     "transform": "proximal -> 0.8, midsubstance -> 0.5, distal -> 0.2 (0 tibia, 1 femur)", "relation": "assumption",
     "assumptions": ["A descriptive tear location is converted to a position along the ligament by a fixed table; it is a "
                     "design assumption, not a measurement."], "review_required": True, "levels": [0]},
]
BY_MAPPING = {m["id"]: m for m in MAPPINGS}

NOT_MAPPED = [
    {"what": "Level 1 surface coatings and nanotexture", "params": "sf.*", "why": "nanometre-scale; not visible on MRI"},
    {"what": "Level 2 microgroove width, depth and spacing", "params": "gr.*", "why": "micrometre-scale; routine knee MRI resolves ~0.3-0.6 mm"},
    {"what": "Level 3 hollow channel lumen, wall and spacing", "params": "ch.*", "why": "micrometre-scale; no implemented method"},
    {"what": "Level 4 braid angle and yarn structure", "params": "scaffold.braid_*", "why": "a manufacturing choice; MRI shows no basis for it"},
    {"what": "Polymer composition and gradients", "params": "*.composition, regions.*.material", "why": "a material choice; not an imaging finding"},
    {"what": "Pretension", "params": "-", "why": "no implemented, justified method maps MRI to pretension"},
    {"what": "Attachment positions in the scaffold frame", "params": "fp.*.center",
     "why": "needs a registration between the MRI and the engine's knee frame, which is not implemented; the distance between "
            "reviewed attachment landmarks can still inform the ligament length"},
]

# --------------------------------------------------------------------------- report extraction
NUM = r"(\d+(?:\.\d+)?)\s*(mm|millimet(?:er|re)s?|cm)"
REPORT_PATTERNS = [
    ("tear_type", "intact", r"\b(?:intact|normal|unremarkable)\s+(?:anterior cruciate ligament|acl)\b|\b(?:anterior cruciate ligament|acl)\s+(?:is\s+)?(?:intact|normal|unremarkable)\b|\bno\s+(?:evidence\s+of\s+)?(?:an?\s+)?(?:acl\s+)?tear\b"),
    ("tear_type", "complete", r"\b(?:complete|full[- ]thickness|full|total)\s+(?:acl\s+)?(?:tear|rupture|disruption)\b|\b(?:acl|anterior cruciate ligament)\s+(?:is\s+)?(?:completely\s+)?(?:ruptured|disrupted)\b"),
    ("tear_type", "partial", r"\b(?:partial(?:[- ]thickness)?|high[- ]grade partial|low[- ]grade partial|incomplete)\s+(?:acl\s+)?(?:tear|rupture)\b"),
    ("tear_bundle", "AM", r"\banteromedial\s+bundle\b|\bAM\s+bundle\b"),
    ("tear_bundle", "PL", r"\bposterolateral\s+bundle\b|\bPL\s+bundle\b"),
    ("tear_location", "proximal", r"\bproximal(?:\s+third)?\b|\bfemoral\s+(?:attachment|insertion|origin|footprint)\b"),
    ("tear_location", "midsubstance", r"\bmid[- ]?substance\b|\bmid[- ]portion\b|\bmiddle\s+third\b"),
    ("tear_location", "distal", r"\bdistal(?:\s+third)?\b|\btibial\s+(?:attachment|insertion|footprint)\b"),
]
REPORT_NUMERIC = [
    ("acl_length", r"(?:acl|ligament)\s+length[^.\d]{0,30}" + NUM),
    ("acl_width", r"(?:acl|ligament)\s+(?:width|thickness)[^.\d]{0,30}" + NUM),
    ("tear_gap", r"(?:gap|retraction|diastasis)[^.\d]{0,30}" + NUM + r"|" + NUM + r"\s+(?:of\s+)?(?:gap|retraction|diastasis)"),
    ("tib_fp_length", r"tibial\s+footprint[^.\d]{0,30}" + NUM),
]


def _mm(v, u):
    return float(v) * (10.0 if u.lower().startswith("cm") else 1.0)


def extract_report(text):
    """Deterministic extraction of explicit statements. Returns findings; never infers beyond the text."""
    text = text or ""
    out = []
    for sent in re.split(r"(?<=[.;\n])\s+", text):
        s = sent.strip()
        if not s:
            continue
        neg = re.search(r"\b(?:no|without|negative for)\b", s, re.I)
        for kind, val, pat in REPORT_PATTERNS:
            m = re.search(pat, s, re.I)
            if not m:
                continue
            if kind != "tear_type" and neg and re.search(r"\bno\b[^.]{0,20}\btear\b", s, re.I):
                continue
            if kind in ("tear_bundle", "tear_location") and not re.search(r"\b(?:tear|rupture|torn|disrupt)", s, re.I):
                continue
            out.append({"kind": kind, "value": val, "units": "", "quote": s[:200], "span": [m.start(), m.end()]})
        for kind, pat in REPORT_NUMERIC:
            m = re.search(pat, s, re.I)
            if m:
                g = [x for x in m.groups() if x]
                out.append({"kind": kind, "value": round(_mm(g[0], g[1]), 2), "units": "mm", "quote": s[:200],
                            "span": [m.start(), m.end()]})
    # an explicit statement wins over a generic one in the same category only if they agree; disagreements are kept
    seen, uniq = set(), []
    for f in out:
        k = (f["kind"], f["value"])
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    for i, f in enumerate(uniq):
        f["id"] = "r%d" % (i + 1)
        f["source_type"] = "report"
        f["certainty"] = "reported"
    return uniq


# --------------------------------------------------------------------------- evidence collection
def collect(manifest):
    """All evidence in a case, with certainty and traceable references."""
    ev = []
    for f in manifest.get("findings", []):
        ev.append(dict(f, ref="report finding %s" % f["id"], label="Report: \"%s\"" % f["quote"][:80]))
    vols = manifest.get("volumes", {})
    anns = manifest.get("annotations", [])
    for a in anns:
        v = vols.get(a.get("volume"), {})
        base = {"ref": "annotation %s" % a["id"], "annotation": a["id"], "volume": a.get("volume"), "volume_name": v.get("name")}
        if a["type"] == "distance" and a.get("kind") in MEASUREMENT_KINDS and a.get("kind") != "other":
            ok = a.get("status") == "reviewed" and a.get("calibrated") and a.get("units") == "mm"
            ev.append(dict(base, kind=a["kind"], value=a.get("value"), units=a.get("units"), source_type="measurement",
                           certainty="measured" if ok else "unreviewed", status=a.get("status"),
                           label="%s (%s)" % (a.get("name") or MEASUREMENT_KINDS[a["kind"]], a.get("status"))))
    lm = {a["kind"]: a for a in anns if a["type"] == "landmark" and a.get("status") == "reviewed" and a.get("calibrated")}
    if "femoral_attachment" in lm and "tibial_attachment" in lm:
        fa, ta = lm["femoral_attachment"], lm["tibial_attachment"]
        if fa["volume"] == ta["volume"]:
            import numpy as np
            d = float(np.linalg.norm(np.subtract(fa["points_world"][0], ta["points_world"][0])))
            ev.append({"kind": "acl_length", "value": round(d, 2), "units": "mm", "source_type": "measurement",
                       "certainty": "measured", "status": "reviewed", "ref": "landmarks %s + %s" % (fa["id"], ta["id"]),
                       "annotation": fa["id"], "volume": fa["volume"], "volume_name": vols.get(fa["volume"], {}).get("name"),
                       "label": "Distance between reviewed attachment landmarks"})
    for mid, mk in manifest.get("masks", {}).items():
        md = mk.get("model") or {}
        t = md.get("tear") or {}
        if mk.get("origin") == "model" and t.get("status"):
            val = "intact" if t["status"] == "intact" else ("complete" if "complete" in (t.get("level") or "") else "partial")
            ev.append({"kind": "tear_type", "value": val, "units": "", "source_type": "model", "certainty": "inferred",
                       "ref": "model output (mask %s)" % mid, "label": "KneeScan tear prediction (probability %s; level reliability %s)"
                       % (t.get("probability"), t.get("level_reliability"))})
    return ev


def _differs(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) > max(1.0, 0.05 * max(abs(a), abs(b)))
    return a != b


def _same(a, b):
    """Exact comparison for current-vs-proposed values (tolerances apply only between evidence sources)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def _width_profile(value):
    spec = D.BY_ID["lig.width_profile"]
    import numpy as np
    base = [float(x) for x in spec["default"]]
    med = float(np.median(np.interp(np.linspace(0.25, 0.75, 101), D.S_KNOTS, base)))
    k = float(value) / med
    return [round(v * k, 2) for v in base], "template profile scaled x%.3f" % k


def propose(manifest, params, provenance=None, scope=None, include_assumptions=False):
    """Build a proposal. Returns {rows, conflicts, missing, not_mapped, changes, summary}.

    rows[].status:
      apply        unambiguous, supported, within limits: applied automatically when the request authorises it
      needs_review assumption mapping or model inference: shown, applied only on explicit confirmation
      conflict     sources disagree with each other or with a user-entered value: the user must choose
      missing      required evidence is not available: current value kept
      blocked      constraint or engine mode prevents the mapping
      unchanged    evidence agrees with the current value
    """
    provenance = provenance or {}
    ev = collect(manifest)
    rows = []
    for mp in MAPPINGS:
        if scope and mp["id"] not in scope:
            continue
        cands = [e for e in ev if e["kind"] == mp["needs"] and e["certainty"] in CERTAINTY_RANK]
        unrev = [e for e in ev if e["kind"] == mp["needs"] and e["certainty"] == "unreviewed"]
        row = {"mapping": mp["id"], "transform": mp["transform"], "relation": mp["relation"], "assumptions": list(mp["assumptions"]),
               "targets": list(mp["targets"].keys()), "levels": mp["levels"], "review_required": mp["review_required"]}
        if not cands:
            row.update(status="missing", reason="No %s available%s." % (MEASUREMENT_KINDS.get(mp["needs"], mp["needs"].replace("_", " ")),
                       " (%d unreviewed measurement(s) exist: review them first)" % len(unrev) if unrev else ""))
            rows.append(row)
            continue
        best_rank = max(CERTAINTY_RANK[e["certainty"]] for e in cands)
        top = [e for e in cands if CERTAINTY_RANK[e["certainty"]] == best_rank]
        others = [e for e in cands if CERTAINTY_RANK[e["certainty"]] < best_rank]
        disagree = [(a, b) for i, a in enumerate(cands) for b in cands[i + 1:] if _differs(a["value"], b["value"])]
        row["sources"] = [{k: e.get(k) for k in ("label", "value", "units", "certainty", "ref", "volume_name", "quote")} for e in cands]
        if disagree:
            row.update(status="conflict", reason="Sources disagree: " + "; ".join(
                "%s says %s%s vs %s says %s%s" % (a["source_type"], a["value"], a.get("units") or "", b["source_type"], b["value"],
                                                   b.get("units") or "") for a, b in disagree[:3]))
            rows.append(row)
            continue
        src = top[0]
        if "requires" in mp and any(params.get(k) != v for k, v in mp["requires"].items()):
            row.update(status="blocked", source=src, reason="Requires %s (currently %s)." % (
                ", ".join("%s = %s" % kv for kv in mp["requires"].items()), ", ".join(str(params.get(k)) for k in mp["requires"])))
            rows.append(row)
            continue
        if "requires_not" in mp and any(params.get(k) == v for k, v in mp["requires_not"].items()):
            row.update(status="blocked", source=src, reason="Not used while %s." % ", ".join("%s = %s" % kv for kv in mp["requires_not"].items()))
            rows.append(row)
            continue
        if "requires_value" in mp:
            eff = dict(params)
            for r2 in rows:
                if r2.get("status") == "apply":
                    eff.update(r2["changes"])
            if any(eff.get(k) != v for k, v in mp["requires_value"].items()):
                row.update(status="blocked", source=src, reason="Applies only when %s." % ", ".join("%s = %s" % kv for kv in mp["requires_value"].items()))
                rows.append(row)
                continue
        changes = {}
        for pid, how in mp["targets"].items():
            if how == "value":
                changes[pid] = src["value"]
            elif how == "location_s":
                changes[pid] = LOCATION_S[src["value"]]
            elif how == "width_profile":
                changes[pid], note = _width_profile(src["value"])
                row["assumptions"].append(note)
            else:
                changes[pid] = how
        row["changes"] = changes
        row["previous"] = {pid: copy.deepcopy(params.get(pid)) for pid in changes}
        row["source"] = {k: src.get(k) for k in ("label", "value", "units", "certainty", "ref", "volume_name", "quote", "source_type")}
        trial = dict(params)
        trial.update(changes)
        errs = [e for e in D.validate_params(trial)[0] if e["param"] in changes]
        user_set = [pid for pid in changes if (provenance.get(pid) or {}).get("source") == "manual edit"
                    and not _same(params.get(pid), changes[pid])]
        if errs:
            row.update(status="blocked", reason="; ".join("%s: %s" % (e["param"], e["message"]) for e in errs))
        elif user_set:
            row.update(status="conflict", reason="You entered %s manually (%s); the evidence gives %s." % (
                ", ".join(user_set), ", ".join(str(params.get(p)) for p in user_set), ", ".join(str(changes[p]) for p in user_set)))
        elif all(_same(params.get(pid), v) for pid, v in changes.items()):
            row.update(status="unchanged", reason="Current value already matches.")
        elif src["certainty"] == "inferred" or mp["review_required"] or src["certainty"] == "assumption":
            row.update(status="apply" if include_assumptions else "needs_review",
                       reason="Model inference or design assumption: confirm before applying." if not include_assumptions else
                       "Applied as a labelled assumption at your request.")
        else:
            row.update(status="apply", reason="Supported mapping from %s evidence." % src["certainty"])
        if others and row["status"] in ("apply", "needs_review", "unchanged"):
            row["also_consistent"] = [o["label"] for o in others]
        rows.append(row)
    changes = {}
    for r in rows:
        if r["status"] == "apply":
            changes.update(r["changes"])
    return {"rows": rows, "changes": changes, "not_mapped": NOT_MAPPED,
            "conflicts": [r for r in rows if r["status"] == "conflict"],
            "summary": {s: sum(1 for r in rows if r["status"] == s) for s in ("apply", "needs_review", "conflict", "missing", "blocked", "unchanged")}}
