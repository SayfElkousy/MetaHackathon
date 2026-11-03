"""Performance/staged-cache and control-metadata regression tests.

- Bone cache keys must be stable when unrelated parameters change (so a
  scaffold-slider edit does not regenerate the bones).
- Staged caching must be invisible: cached builds equal fresh builds.
- Every registry parameter must carry scope/level/help metadata from the
  same single source used for labels, units, bounds and validation.
"""
import copy

import numpy as np

from acl_engine.v2 import design as D
from acl_engine.v2.bones import ProceduralBone
from acl_engine.v2.scene import build_scene, clear_stage_caches


def _geom_hash(sc):
    import hashlib
    h = hashlib.sha256()
    for o in sc["objects"]:
        if o["kind"] == "mesh":
            h.update(np.round(o["V"], 3).astype(np.float32).tobytes())
        else:
            for ln in o["lines"]:
                h.update(np.round(ln["P"], 4).astype(np.float32).tobytes())
    return h.hexdigest()


def test_bone_key_ignores_unrelated_params():
    p0 = D.default_params()
    p1 = dict(p0)
    p1["scaffold.yarn_diameter_mm"] = 0.7  # scaffold-only edit
    p1["injury.gap_mm"] = 7.0  # injury-only edit
    for which in ("femur", "tibia"):
        a = ProceduralBone(which, p0, 0.0).key()
        b = ProceduralBone(which, p1, 0.0).key()
        assert a == b, (which, a, b)
    # ... but bone params must change the key
    p2 = dict(p0)
    p2["bone.notch_width_mm"] = 22.0
    assert ProceduralBone("femur", p0, 0.0).key() != ProceduralBone("femur", p2, 0.0).key()


def test_cached_build_matches_fresh_build():
    base = D.new_state()
    clear_stage_caches()
    b0 = build_scene(base, "full")
    assert b0["ok"]
    g0 = _geom_hash(b0)
    # perturb one parameter from each area; cached rebuilds must equal fresh ones
    probes = [
        {"scaffold.yarn_diameter_mm": 0.7},
        {"injury.gap_mm": 7.0},
        {"lig.size_scale": 1.1},
        {"native.fascicle_count": 40},
        {"scaffold.braid_angle_deg": 30.0},
        {"injury.type": "partial", "injury.involvement": "AM"},
        {"fit.partition": "per_bundle"},
    ]
    for probe in probes:
        pid = ",".join(probe)
        st = copy.deepcopy(base)
        st["params"].update(probe)
        cached = build_scene(st, "full")  # may reuse stages
        assert cached["ok"], pid
        clear_stage_caches()
        fresh = build_scene(st, "full")
        assert fresh["ok"], pid
        assert _geom_hash(cached) == _geom_hash(fresh), pid


def test_preview_reuses_final_where_valid_and_stays_light():
    st = D.new_state()
    clear_stage_caches()
    full = build_scene(st, "full")
    prev = build_scene(st, "preview")
    assert full["ok"] and prev["ok"]
    # same design hash (display/quality never change the design)
    assert full["design_hash"] == prev["design_hash"]
    # preview carries per-object content keys for client-side reuse
    assert all("ckey" in o for o in prev["objects"])
    assert all("ckey" in o for o in full["objects"])


def test_registry_metadata_complete():
    reg = D.registry_for_ui()
    groups = {g["id"] for g in reg["groups"]}
    # Anatomy and Pose groups + the six scaffold levels (the old 'fit' / 'materials' groups were
    # redistributed so that every scaffold control has exactly one owning level)
    for want in ("anatomy", "am", "pl", "injury", "fibers", "rotation", "lvl1", "lvl2", "lvl3", "lvl4", "lvl5", "lvl6"):
        assert want in groups, want
    scopes = {"whole ACL", "AM region", "PL region", "injury", "scaffold", "scaffold: AM-like region", "scaffold: PL-like region",
              "scaffold: tibial end", "scaffold: femoral end"}
    for p in reg["params"]:
        assert p["group"] in groups, p["id"]
        assert p.get("scope") in scopes, p["id"]
        assert p.get("level") in ("basic", "advanced"), p["id"]
        h = p.get("help") or {}
        for k in ("what", "increase", "decrease", "look", "depends", "guide"):
            assert h.get(k), (p["id"], k)
