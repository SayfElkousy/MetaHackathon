"""Scene builder: design state -> named objects, checks, metrics, section data.

Every object carries a provenance status for display and export:
  observed  - comes from patient data (imported mesh / measured values)
  assumed   - generic template or synthetic demonstration geometry
  estimated - reconstruction (e.g. intact reference)
  proposed  - the scaffold design
"""
import hashlib
import json
import math
import time
from collections import OrderedDict
import numpy as np
from scipy.spatial import cKDTree
from acl_engine.v2 import mesher
from acl_engine.v2.bones import ProceduralBone, MeshBone, cached_mesh, load_mesh_file
from acl_engine.v2.design import (validate_params, design_hash, PACKING_LIMIT, migrate_params,
                                  fascicle_sizing, MATERIALS, REGION_IDS, DISPLAY_DEFAULTS, coerce_types,
                                  BONE_PARAMS, LIG_PARAMS, TISSUE_PARAMS, SCAFFOLD_PARAMS, LAYER_PARAMS,
                                  inactive_params, LEVELS)
from acl_engine.v2 import fit as fitmod
from acl_engine.v2 import rotation as rotmod
from acl_engine.v2 import sections as secmod
from acl_engine.v2.ligament import Ligament, profile_eval, sunflower, rot2
from acl_engine.v2.tissue import Injury, NativeTissue, envelope_samples, intact_loft
from acl_engine.v2.scaffold import Scaffold
from acl_engine.v2.layered import LayeredScaffold

BUNDLE_NAMES = ["AM", "PL"]
_BONE_FILE_CACHE = {}
_MESHBONE_CACHE = OrderedDict()

# Staged build caches: each stage rebuilds only when its own explicit
# parameter subset (or an upstream stage) changes. Bounded LRU so repeated
# slider edits cannot accumulate memory.
_STAGE_CACHES = {"lig": OrderedDict(), "tissue": OrderedDict(),
                 "surface": OrderedDict(), "scaffold": OrderedDict(),
                 "rotation": OrderedDict(), "layered": OrderedDict()}
_STAGE_CAP = 4


def clear_stage_caches():
    for c in _STAGE_CACHES.values():
        c.clear()
    _MESHBONE_CACHE.clear()


def _subset_key(p, keys, extra=""):
    d = {k: p.get(k) for k in keys}
    return hashlib.sha1((json.dumps(d, sort_keys=True, default=str) + "|" + str(extra)).encode()).hexdigest()[:16]


def _assets_key(assets):
    ids = {}
    for k in ("femur_mesh", "tibia_mesh"):
        a = (assets or {}).get(k)
        ids[k] = [a.get("path"), str(a.get("transform")), a.get("units_to_mm"), a.get("mirror_x")] if a else None
    ids["stumps"] = [s.get("path") for s in (assets or {}).get("stump_meshes", []) or []]
    ids["injury"] = [(assets or {}).get("injury_mask"), (assets or {}).get("injury_table")]
    ids["ref"] = (assets or {}).get("reference_source")
    return hashlib.sha1(json.dumps(ids, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _stage_get(name, key):
    c = _STAGE_CACHES[name]
    if key in c:
        c.move_to_end(key)
        return c[key]
    return None


def _stage_put(name, key, value):
    c = _STAGE_CACHES[name]
    c[key] = value
    c.move_to_end(key)
    while len(c) > _STAGE_CAP:
        c.popitem(last=False)


def _status_from_prov(prov, pids, default="assumed"):
    sts = [prov.get(p, {}).get("status") for p in pids]
    if sts and all(s in ("measured", "observed") for s in sts):
        return "observed"
    if any(s in ("measured", "inferred", "observed") for s in sts):
        return "partly observed"
    return default


def _load_bone(asset, which, flexion):
    if not asset:
        return None
    key = (asset["path"], str(asset.get("transform")), which)
    if key not in _BONE_FILE_CACHE:
        V, F = load_mesh_file(asset["path"])
        V = V * float(asset.get("units_to_mm", 1.0))
        Tm = np.asarray(asset.get("transform", np.eye(4)), float)
        V = V @ Tm[:3, :3].T + Tm[:3, 3]
        if asset.get("mirror_x"):
            V[:, 0] *= -1
            F = F[:, ::-1]
        _BONE_FILE_CACHE[key] = (V, F)
    V, F = _BONE_FILE_CACHE[key]
    mk = (key, round(float(flexion), 3))
    if mk not in _MESHBONE_CACHE:
        if len(_MESHBONE_CACHE) > 8:
            _MESHBONE_CACHE.popitem(last=False)
        _MESHBONE_CACHE[mk] = MeshBone(which, V, F, flexion, source=asset)
    return _MESHBONE_CACHE[mk]


def geometry_hash(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.round(np.asarray(a, float), 4).astype(np.float32).tobytes())
    return h.hexdigest()[:16]


def build_scene(state, quality="full"):
    t0 = time.time()
    tm = {}
    params = dict(state["params"])
    migration = migrate_params(params)
    p = coerce_types(params)
    prov = state.get("provenance", {})
    disp = dict(DISPLAY_DEFAULTS, **(state.get("display") or {}))
    errs, warns = validate_params(p)
    if errs:
        return {"ok": False, "errors": errs, "warnings": warns}
    assets = state.get("assets") or {}
    flex = float(p["knee.flexion_deg"])
    side = p["case.side"]
    akey = _assets_key(assets)
    hits = {}
    # ---- bones (cached by bone-only keys: scaffold/injury edits reuse them)
    t = time.time()
    fem = _load_bone(assets.get("femur_mesh"), "femur", flex) or ProceduralBone("femur", p, flex)
    tib = _load_bone(assets.get("tibia_mesh"), "tibia", 0.0) or ProceduralBone("tibia", p, 0.0)
    objs = []
    hb = 0.9 if quality == "full" else 1.6
    hcart = (0.7, 0.6) if quality == "full" else (1.2, 1.1)
    for bone, oid, lab in ((fem, "bone_femur", "Distal femur"), (tib, "bone_tibia", "Proximal tibia")):
        V, F = cached_mesh(bone, hb)
        objs.append({"id": oid, "label": lab, "category": "bone", "kind": "mesh", "V": V, "F": F,
                     "status": "observed" if bone.kind == "mesh" else "assumed",
                     "note": ("patient surface mesh" if bone.kind == "mesh" else
                              "procedural approximation of adult anatomy, not patient data"),
                     "cache_key": bone.key() + "_%s" % hb, "ckey": "bone:%s:%s" % (bone.key(), hb)})
    if fem.kind == "procedural":
        cb = ProceduralBone("femoral_cartilage", p, flex)
        V, F = cached_mesh(cb, hcart[0])
        objs.append({"id": "cartilage_femoral", "label": "Femoral cartilage", "category": "cartilage", "kind": "mesh",
                     "V": V, "F": F, "status": "assumed", "note": "uniform-thickness approximation",
                     "cache_key": cb.key() + "_%s" % hcart[0], "ckey": "bone:%s:%s" % (cb.key(), hcart[0])})
    if tib.kind == "procedural":
        cb = ProceduralBone("tibial_cartilage", p, 0.0)
        V, F = cached_mesh(cb, hcart[1])
        objs.append({"id": "cartilage_tibial", "label": "Tibial cartilage", "category": "cartilage", "kind": "mesh",
                     "V": V, "F": F, "status": "assumed", "note": "uniform-thickness approximation",
                     "cache_key": cb.key() + "_%s" % hcart[1], "ckey": "bone:%s:%s" % (cb.key(), hcart[1])})
    tm["bones"] = time.time() - t
    bone_keys = fem.key() + tib.key()
    # ---- ligament + footprints (cached by ligament subset + bones + quality)
    t = time.time()
    lig_key = _subset_key(p, LIG_PARAMS, bone_keys + quality + akey)
    lig_entry = _stage_get("lig", lig_key)
    if lig_entry is None:
        lig = Ligament(p, fem, tib, quality)
        fp_meshes = {}
        for sd in ("tib", "fem"):
            fp = lig.fp[sd]
            rot = lig.align[sd]
            th_end = float(lig.theta_b[0 if sd == "tib" else -1])
            bf = lambda f, rot=rot, th=th_end: lig.bundle_of(rot2(rot2(f, -rot), -th))
            V, F, bid = fp.mesh(bundle_fn=bf)
            fp_meshes[sd] = (V, F, bid, fp.shape, fp.area())
        a = tib.project(np.array([[0.0, -21.0, -2.0]]))[0]
        b_local = np.array([[-9.0, 6.0, 20.0]])
        b = fem.project(fem.to_world(b_local) if hasattr(fem, "to_world") else b_local)[0]
        mid = 0.5 * (a + b) + np.array([0, -4.0, 0])
        tt = np.linspace(0, 1, 40)[:, None]
        pcl = (1 - tt) ** 2 * a + 2 * (1 - tt) * tt * mid + tt ** 2 * b
        pcl_mesh = mesher.tube_mesh(pcl, 5.0, nsec=16)
        stump_meshes = []
        for k, sm in enumerate(assets.get("stump_meshes") or []):
            V, F = load_mesh_file(sm["path"])
            Tm = np.asarray(sm.get("transform", np.eye(4)), float)
            V = V @ Tm[:3, :3].T + Tm[:3, 3]
            stump_meshes.append((k, sm.get("label", k), V, F))
        lig_entry = (lig, fp_meshes, pcl_mesh, stump_meshes)
        _stage_put("lig", lig_key, lig_entry)
        hits["lig"] = False
    else:
        hits["lig"] = True
    lig, fp_meshes, pcl_mesh, stump_meshes = lig_entry
    fp_status = {"tib": _status_from_prov(prov, ["fp.tib.center", "fp.tib.length_mm", "fp.tib.width_mm"]),
                 "fem": _status_from_prov(prov, ["fp.fem.center", "fp.fem.length_mm", "fp.fem.width_mm"])}
    for sd, oid, lab in (("tib", "footprint_tibial", "Tibial footprint"), ("fem", "footprint_femoral", "Femoral footprint")):
        V, F, bid, shape, area = fp_meshes[sd]
        objs.append({"id": oid, "label": lab, "category": "footprint", "kind": "mesh", "V": V, "F": F,
                     "attr": {"bundle": bid}, "status": fp_status[sd],
                     "note": "%s, %.0f mm2 surface area" % (shape, area), "ckey": "lig:" + lig_key})
    V, F = pcl_mesh
    objs.append({"id": "landmark_pcl", "label": "PCL (landmark)", "category": "landmark", "kind": "mesh", "V": V, "F": F,
                 "status": "assumed", "note": "schematic landmark for orientation only", "ckey": "lig:" + lig_key})
    for k, lab, V, F in stump_meshes:
        objs.append({"id": "observed_stump_%d" % k, "label": "Observed stump surface: %s" % lab, "category": "observed",
                     "kind": "mesh", "V": V, "F": F, "status": "observed",
                     "note": "segmentation surface from the MRI model", "ckey": "lig:" + lig_key})
    tm["ligament"] = time.time() - t
    # ---- injury + tissue (cached by tissue subset + ligament + quality)
    t = time.time()
    tis_key = _subset_key(p, TISSUE_PARAMS, lig_key + akey)
    tis_entry = _stage_get("tissue", tis_key)
    if tis_entry is None:
        inj = Injury(p, lig, assets)
        nat = NativeTissue(p, lig, inj, disp)
        fasc = nat.fascicles()
        tis_entry = (inj, nat, fasc)
        _stage_put("tissue", tis_key, tis_entry)
        hits["tissue"] = False
    else:
        hits["tissue"] = True
    inj, nat, fasc = tis_entry
    tm["fascicles"] = time.time() - t
    t = time.time()
    h = 0.35 if quality == "full" else 0.7
    surf_key = _subset_key(p, TISSUE_PARAMS | {"bundle.interface_uncertainty_mm"}, lig_key + quality + akey)
    surf_entry = _stage_get("surface", surf_key)
    if surf_entry is None:
        S, M, P, D = envelope_samples(lig, inj, quality)
        Vi, Fi = intact_loft(lig)
        surf_meshes = {"intact": (Vi, Fi)}
        if inj.active:
            Vr, Fr = mesher.mesh_from_points(P[~D], h=h)
            parts = mesher.split_components(Vr, Fr, min_faces=60)
            tc, fc = lig.fp["tib"].center, lig.fp["fem"].center
            remnants = []
            for k, (V, F) in enumerate(parts):
                dt = np.linalg.norm(V - tc, axis=1).min()
                df = np.linalg.norm(V - fc, axis=1).min()
                if dt < 2.5 and df < 2.5:
                    oid, lab = "retained_tissue", "Retained tissue (continuous)"
                elif dt < df:
                    oid, lab = "retained_tibial_stump", "Tibial remnant"
                else:
                    oid, lab = "retained_femoral_stump", "Femoral remnant"
                if any(r[0] == oid for r in remnants):
                    oid += "_%d" % k
                remnants.append((oid, lab, V, F))
            surf_meshes["remnants"] = remnants
            if D.any():
                surf_meshes["injury"] = mesher.mesh_from_points(P[D], h=h)
        bundles = []
        for b in (0, 1):
            Vb, Fb_ = secmod.ring_loft(lig, secmod.bundle_ring(lig, b))
            bundles.append((Vb, Fb_))
        unc = float(p["bundle.interface_uncertainty_mm"])
        surf_meshes["bundles"] = bundles
        surf_meshes["interface"] = secmod.interface_band(lig, unc)
        surf_meshes["paths"] = [lig.bundle_centerline(b) for b in (0, 1)]
        surf_entry = (S, M, P, D, surf_meshes)
        _stage_put("surface", surf_key, surf_entry)
        hits["surface"] = False
    else:
        hits["surface"] = True
    S, M, P, D, surf_meshes = surf_entry
    unc = float(p["bundle.interface_uncertainty_mm"])
    inj_status = "observed" if (inj.table or inj.mask is not None) else _status_from_prov(prov, ["injury.center_s", "injury.gap_mm"], "assumed")
    Vi, Fi = surf_meshes["intact"]
    objs.append({"id": "intact_reference", "label": "Intact ACL (reference estimate)", "category": "reference", "kind": "mesh",
                 "V": Vi, "F": Fi, "status": "estimated", "ckey": "surf:" + surf_key,
                 "note": "envelope from the anatomy parameters%s" % (" (contralateral)" if assets.get("reference_source") else "")})
    remnant_info = []
    if inj.active:
        for oid, lab, V, F in surf_meshes.get("remnants", []):
            objs.append({"id": oid, "label": lab, "category": "retained", "kind": "mesh", "V": V, "F": F, "status": inj_status,
                         "note": "remnant surface derived from the injury region", "ckey": "surf:" + surf_key})
            remnant_info.append((oid, V))
        if "injury" in surf_meshes:
            Vd, Fd = surf_meshes["injury"]
            objs.append({"id": "injury_region", "label": "Missing / damaged tissue", "category": "injury", "kind": "mesh",
                         "V": Vd, "F": Fd, "status": inj_status, "note": "volume of the intact estimate inside the injury region",
                         "ckey": "surf:" + surf_key})
    else:
        objs.append({"id": "retained_tissue", "label": "Native ACL (intact)", "category": "retained", "kind": "mesh",
                     "V": Vi, "F": Fi, "status": _status_from_prov(prov, ["lig.width_profile"], "assumed"), "note": "no injury",
                     "ckey": "surf:" + surf_key})
    # AM/PL bundle regions: exact lofts of each bundle's material outline (the same
    # section model the scaffold is fitted to), so sections and exports agree
    bstat = _status_from_prov(prov, ["bundle.am_fraction", "bundle.boundary_angle_deg"], "assumed")
    for b in (0, 1):
        Vb, Fb_ = surf_meshes["bundles"][b]
        objs.append({"id": "bundle_%s_reference" % BUNDLE_NAMES[b].lower(), "label": "%s bundle region (intact)" % BUNDLE_NAMES[b],
                     "category": "bundle", "kind": "mesh", "V": Vb, "F": Fb_, "status": bstat, "ckey": "surf:" + surf_key,
                     "note": "%s area at mid-length %.1f mm2 (%.0f%% of the section)" % (BUNDLE_NAMES[b], lig.area_bundle[lig.s.size // 2, b],
                                                                                      100 * lig.area_bundle[lig.s.size // 2, b] / lig.area[lig.s.size // 2])})
    Vu, Fu = surf_meshes["interface"]
    objs.append({"id": "bundle_interface", "label": "AM/PL boundary (uncertainty band)", "category": "interface", "kind": "mesh",
                 "V": Vu, "F": Fu, "status": bstat, "ckey": "surf:" + surf_key,
                 "note": "the true AM/PL boundary is expected within +/- %.2f mm of this surface; it is not sharply visible on MRI" % unc})
    paths = surf_meshes["paths"]
    objs.append({"id": "bundle_paths", "label": "AM / PL bundle paths", "category": "guide", "kind": "lines",
                 "lines": [{"P": paths[b], "r": np.full(paths[b].shape[0], 0.06), "bundle": b} for b in (0, 1)],
                 "status": "estimated", "note": "centroid path of each bundle region, attachment subregion to subregion",
                 "ckey": "surf:" + surf_key})
    tm["surfaces"] = time.time() - t
    # ---- fascicles and fibers (polyline sets)
    t = time.time()
    damage = profile_eval(p["injury.damage_profile"], lig.s)
    fas_lines = []
    for fa in fasc:
        for (a0, a1) in _runs(~fa["damaged"]):
            if a1 - a0 < 2:
                continue
            fas_lines.append({"P": fa["P"][a0:a1], "r": fa["r"][a0:a1], "fascicle": fa["id"], "bundle": fa["bundle"],
                              "bridging": fa["bridging"], "damage": damage[a0:a1]})
    disp_fib_key = "%s:%s" % (disp["native_fibers_per_fascicle"], disp["crimp_exaggeration"])
    objs.append({"id": "native_fascicles", "label": "Retained fascicle groups", "category": "retained_structure",
                 "kind": "tubes", "lines": fas_lines, "status": inj_status if inj.active else "assumed",
                 "ckey": "tis:%s:%s" % (tis_key, disp_fib_key),
                 "note": "%d representative fascicle groups, d = %.2f mm" % (nat.n_fasc, nat.d_fasc)})
    n_rep = int(disp["native_fibers_per_fascicle"])
    if quality != "full":
        n_rep = max(1, min(3, n_rep))  # preview: fewer representative fibers (display only)
    fib = nat.fibers(fasc, n_rep, float(disp["crimp_exaggeration"]))
    fib_lines = []
    for fr in fib:
        keep = ~fr["damaged"]
        for (a0, a1) in _runs(keep):
            if a1 - a0 < 2:
                continue
            for k in range(fr["P"].shape[0]):
                fib_lines.append({"P": fr["P"][k, a0:a1], "r": np.full(a1 - a0, fr["r"][k]), "fascicle": fr["fascicle"],
                                  "bundle": fr["bundle"], "fiber": k})
    objs.append({"id": "native_fibers", "label": "Representative fibers (native)", "category": "retained_structure",
                 "kind": "lines", "lines": fib_lines, "status": "assumed",
                 "ckey": "tis:%s:%s:%s" % (tis_key, disp_fib_key, quality),
                 "note": "%d shown per fascicle; crimp drawn x%g (display only)" % (n_rep, disp["crimp_exaggeration"])})
    # centreline + control handles
    objs.append({"id": "centerline", "label": "Centreline", "category": "guide", "kind": "lines",
                 "lines": [{"P": lig.C, "r": np.full(lig.C.shape[0], 0.08)}], "status": "estimated", "note": "",
                 "ckey": "lig:" + lig_key})
    tm["fibers"] = time.time() - t
    # ---- scaffold (cached by scaffold subset + ligament + tissue + bones + quality)
    t = time.time()
    scaf_key = _subset_key(p, SCAFFOLD_PARAMS, lig_key + tis_key + bone_keys + quality + akey)
    scaf_entry = _stage_get("scaffold", scaf_key)
    if scaf_entry is None:
        scf = Scaffold(p, lig, inj, nat, fasc, quality, bones=[fem, tib])
        yarns, clear, ssum, fitr = [], {}, {}, {}
        if scf.active:
            yarns = scf.yarns([fem, tib])
            clear = scf.clearance(yarns, fasc)
            fitr = fitmod.evaluate(lig, inj, scf, yarns, clear, [fem, tib], fasc)
            ssum = scf.summary(yarns)
        # repair-space volumes and region domains (depend on the scaffold units)
        scaf_meshes = []
        if scf.active:
            L = lig.L
            for u in scf.units:
                suffix = "" if len(scf.units) == 1 else "_" + u.id.lower()
                ring = _unit_ring(lig, u)
                V, F = _unit_volume(lig, u, ring, u.sig_a, u.sig_b, h)
                if F.shape[0]:
                    scaf_meshes.append(("repair_space%s" % suffix,
                                        "Repair space: %s (permitted scaffold volume)" % u.label,
                                        "repair_space", V, F,
                                        "affected %s region from %.1f to %.1f mm (damaged interval + permitted overlap %.1f/%.1f mm)%s"
                                        % ("bundle" if u.iface_side is not None else "section", u.sig_a, u.sig_b, u.ov_t, u.ov_f,
                                           "; reaches bone" if (u.bone_tib or u.bone_fem) else ""), None))
                bounds = {"tibial_interface": (u.sig_a, u.b1), "bridge": (u.b1, u.b2), "femoral_interface": (u.b2, u.sig_b)}
                for rname, (a, b) in bounds.items():
                    a, b = max(a, u.sig_a), min(b, u.sig_b)
                    if b - a < 0.3:
                        continue
                    V, F = _unit_volume(lig, u, ring, a, b, h)
                    if not F.shape[0]:
                        continue
                    mat = scf.reg[rname]["material"]
                    scaf_meshes.append(("scaffold_region_%s%s" % (rname, suffix),
                                        "Scaffold region domain: %s (%s)%s" % (rname.replace("_", " "), u.region_kind[rname].replace("_", " "),
                                                                             "" if not suffix else " - " + u.label),
                                        "scaffold_region", V, F,
                                        "%s; the part of the repair space assigned to this material region (composition is metadata)" % MATERIALS[mat]["label"],
                                        {"region": rname, "region_kind": u.region_kind[rname], "material": mat, "unit": u.id,
                                         "composition": scf.reg[rname]["composition"], "color": MATERIALS[mat]["color"]}))
            if fitr.get("violations"):
                Vv, Fv = _markers([v["P"] for v in fitr["violations"]], 0.22)
                scaf_meshes.append(("fit_violations", "Fit violations (%d marked)" % len(fitr["violations"]),
                                    "violation", Vv, Fv,
                                    "locations where a strand leaves its repair space, enters bone or another strand, or overlaps retained tissue", None))
        scaf_entry = (scf, yarns, clear, ssum, fitr, scaf_meshes)
        _stage_put("scaffold", scaf_key, scaf_entry)
        hits["scaffold"] = False
    else:
        hits["scaffold"] = True
    scf, yarns, clear, ssum, fitr, scaf_meshes = scaf_entry
    if scf.active:
        anch_t = any(u.bone_tib for u in scf.units)
        anch_f = any(u.bone_fem for u in scf.units)
        for o in objs:
            if (o["id"] == "retained_tibial_stump" and anch_t) or (o["id"] == "retained_femoral_stump" and anch_f):
                o["label"] += " (shorter than the bone-reach limit: occupied by the repair, assumed debrided)"
        bad = set(clear.get("yarn_bad_ids", []))
        unit_ids = [u.id for u in scf.units]
        ylines = [{"P": y["P"], "r": y["r"], "yarn": y["id"], "family": y["family"], "region": y["region"],
                   "collide": y["id"] in bad, "viol": y.get("viol"), "unit": unit_ids.index(y["unit"]),
                   "layer": int(y.get("layer", 0))} for y in yarns]
        objs.append({"id": "scaffold_yarns", "label": "Scaffold yarns", "category": "scaffold", "kind": "tubes",
                     "lines": ylines, "status": "proposed", "units": unit_ids, "ckey": "scaf:" + scaf_key,
                     "note": "%s, %d yarns in %d repair unit%s" % (p["scaffold.arch"], len(yarns), len(unit_ids), "s" if len(unit_ids) > 1 else "")})
        n_srep = int(disp["scaffold_fibers_per_yarn"])
        if quality != "full":
            n_srep = max(1, min(3, n_srep))  # preview: fewer representative fibers (display only)
        sf = scf.fibers(yarns, n_srep, float(disp["crimp_exaggeration"]))
        sl = []
        for fr in sf:
            for k in range(fr["P"].shape[0]):
                sl.append({"P": fr["P"][k], "r": np.full(fr["P"].shape[1], fr["r"][k]), "yarn": fr["yarn"],
                           "fiber": k, "region": fr["region"]})
        objs.append({"id": "scaffold_fibers", "label": "Representative fibers (scaffold)", "category": "scaffold",
                     "kind": "lines", "lines": sl, "status": "proposed",
                     "ckey": "scaf:%s:%s:%s" % (scaf_key, disp["scaffold_fibers_per_yarn"], disp["crimp_exaggeration"]),
                     "note": "%d shown per yarn" % n_srep})
        # repair-space volumes and region domains come from the cached scaffold stage
        _scaf_note = None
        for oid, lab, cat, V, F, note, extra in scaf_meshes:
            o = {"id": oid, "label": lab, "category": cat, "kind": "mesh", "V": V, "F": F,
                 "status": "estimated" if cat == "repair_space" else "proposed",
                 "note": note, "ckey": "scaf:" + scaf_key}
            if extra:
                o.update(extra)
            objs.append(o)
    tm["scaffold"] = time.time() - t
    # ---- six-level layered construct (cached by its own controls + ligament + bones + quality)
    t = time.time()
    lay_key = _subset_key(p, LAYER_PARAMS, lig_key + bone_keys + quality + akey)
    lay = _stage_get("layered", lay_key)
    if lay is None:
        lay = LayeredScaffold(p, lig, [fem, tib], quality, inj, MATERIALS, prov)
        _stage_put("layered", lay_key, lay)
        hits["layered"] = False
    else:
        hits["layered"] = True
    if lay.active:
        for o in lay.objects:
            o2 = dict(o)                        # per-build copies: mirroring must not touch the cache
            if "lines" in o2:
                o2["lines"] = [dict(ln) for ln in o["lines"]]
            if "markers" in o2:
                o2["markers"] = [dict(mk, P=list(mk["P"])) for mk in o["markers"]]
            o2["ckey"] = "lay:%s:%s" % (lay_key, o["id"])
            objs.append(o2)
        for o in objs:
            if o["category"] == "retained" and inj.active:
                o["label"] += " (reference only: the layered construct spans footprint to footprint)"
    tm["layered"] = time.time() - t
    t = time.time()
    # validated on the generated paths at TRUE scale (display crimp magnification excluded)
    rotr = {"limit_deg": rotmod.LIMIT_DEG, "tolerance_deg": rotmod.TOL_DEG, "rows": [], "note": "measured on full builds"}
    if quality == "full":
        rot_key = scaf_key if scf.active else tis_key + lig_key
        rotr = _stage_get("rotation", rot_key)
        if rotr is None:
            rotr = rotmod.evaluate(p, lig, nat, fasc, scf if scf.active else None, yarns,
                                   scf.fibers(yarns, 7, 1.0) if scf.active and yarns else None,
                                   native_fibers=nat.fibers(fasc, 7, 1.0, n_samples=lig.s.size))
            _stage_put("rotation", rot_key, rotr)
            hits["rotation"] = False
        else:
            hits["rotation"] = True
    tm["rotation"] = time.time() - t
    # ---- checks & metrics
    t = time.time()
    checks = run_checks(p, prov, lig, inj, nat, fasc, scf, yarns, clear, objs, fem, tib, S, P, D, warns, quality, fitr)
    checks[0:0] = rotation_checks(rotr)
    if lay.active:
        checks[0:0] = layer_checks(lay, inj)
    for m in migration:
        checks.append(_chk("migration", "Design migrated from an older version", "info", None, None, "", [], m, "input"))
    tm["checks"] = time.time() - t
    metrics = collect_metrics(p, lig, inj, nat, scf, ssum, clear, remnant_info)
    if fitr:
        metrics["fit"] = fitr
    metrics["rotation"] = rotr
    if lay.active:
        metrics["layered"] = lay.report()
        metrics["layered"]["footprint_source"] = footprint_sources(p, prov)
    fails = [c for c in checks if c["status"] == "fail"]
    candidates = []
    if fails and scf.active:
        candidates = _dedupe(scf.resolutions + _generic_candidates(p, scf, yarns, fails))
    validity = {"valid": not fails, "fail": len(fails), "warn": sum(1 for c in checks if c["status"] == "warn"),
                "failed": [{"id": c["id"], "label": c["label"], "message": c["message"], "params": c["params"]} for c in fails]}
    arrays = [o["V"] for o in objs if o["kind"] == "mesh" and o["category"] not in ("bone", "cartilage")]
    arrays += [ln["P"] for o in objs if o["kind"] in ("tubes",) for ln in o["lines"]]
    ghash = geometry_hash(arrays)
    tm["total"] = time.time() - t0
    tm["cache_hits"] = hits
    handles = {"ctrl": lig.ctrl[1:3].tolist(), "fp_tib": lig.fp["tib"].center.tolist(), "fp_fem": lig.fp["fem"].center.tolist(),
               "ctrl_base": [(lig.ctrl[k] - np.asarray(p["lig.ctrl_offsets"][k - 1])).tolist() for k in (1, 2)]}
    if side == "left":
        for o in objs:
            if o["kind"] == "mesh":
                o["V"] = o["V"] * np.array([-1, 1, 1])
                o["F"] = o["F"][:, ::-1]
                for mk in o.get("markers") or []:
                    mk["P"] = [-mk["P"][0], mk["P"][1], mk["P"][2]]
            else:
                for ln in o["lines"]:
                    ln["P"] = ln["P"] * np.array([-1, 1, 1])
                    if "u" in ln:
                        ln["u"] = ln["u"] * np.array([-1, 1, 1])
        handles = {k: (np.asarray(v) * np.array([-1, 1, 1])).tolist() for k, v in handles.items()}
    if quality == "full":
        section = secmod.section(objs, lig, disp.get("section_kind", "ligament"), _section_pos(disp), side,
                                 unc, (fitr or {}).get("violations"))
    else:
        section = None  # previews skip the section cut; the viewer keeps the previous one
    return {"ok": True, "objects": objs, "checks": checks, "metrics": metrics, "section": section, "validity": validity,
            "resolutions": [], "resolution_candidates": candidates, "lig": lig,
            "_ctx": (p, lig, inj, nat, fasc, quality, fem, tib),
            "warnings": warns, "timings_s": tm, "design_hash": design_hash(state), "geometry_hash": ghash,
            "handles": handles, "side": side, "quality": quality, "inactive": inactive_params(p), "_lay": lay,
            "frame": "mm; +X lateral (right knee), +Y anterior, +Z proximal; origin tibial plateau centre"}


def layer_checks(lay, inj):
    """Conflicts of the six-level construct as checks (with owner level, involved levels
    and concrete fixes), plus the scope notes."""
    out = []
    for c in lay.conflicts:
        ch = _chk(c["id"], "L%d %s: %s" % (c["level"], lay_level_name(c["level"]), c["title"]), c["status"], c["value"], c["threshold"],
                  c["units"], c["params"], c["explain"], "layer")
        ch.update(level=c["level"], levels=c["levels"], fixes=c["fixes"])
        out.append(ch)
    if not any(c["status"] == "fail" for c in lay.conflicts):
        out.append(dict(_chk("layer_fit", "Layered construct: every level fits its neighbours", "pass", None, None, "", [],
                             "channels inside their envelopes, braids inside the sheath clearance, grooves inside the walls, "
                             "flare bases inside the footprints (measured on the generated geometry)", "layer"), level=None, levels=[]))
    if inj.active:
        out.append(dict(_chk("layer_remnants", "Retained remnants and the layered construct", "info", None, None, "", ["scaffold.arch"],
                             "The layered construct spans footprint to footprint; retained remnants are drawn for reference. "
                             "Remnant handling (debridement or preservation) is not modelled.", "layer"), level=5, levels=[5]))
    return out


def lay_level_name(n):
    return {L["n"]: L["name"] for L in LEVELS}.get(n, "")


def footprint_sources(p, prov):
    """Where each scaffold footprint comes from: imaging, manual placement or a default template."""
    from acl_engine.v2.design import BY_ID
    out = {}
    for side in ("tib", "fem"):
        st = [prov.get("fp.%s.%s" % (side, f), {}).get("status") for f in ("center", "length_mm", "width_mm", "shape", "orientation_deg")]
        manual_end = any(p["end.%s.%s" % (side, f)] != BY_ID["end.%s.%s" % (side, f)]["default"]
                         for f in ("offset_long_mm", "offset_short_mm", "rotation_deg", "outline", "size_mode"))
        if manual_end or "override" in st:
            src = "manual placement"
        elif any(x in ("measured", "observed", "inferred") for x in st):
            src = "imaging"
        else:
            src = "default template"
        out[side] = {"source": src, "anatomical_status": [x for x in st if x],
                     "note": {"imaging": "anatomical footprint from the MRI-analysis import",
                              "manual placement": "moved, resized or reshaped by hand (Anatomy and Pose or Level 6)",
                              "default template": "literature template footprint (not patient data)"}[src]}
    return out


def verify_layer_fixes(scene, limit=12):
    """Rebuild only the layered construct for each fix of a failing conflict and report
    whether that conflict is resolved and which new failures appear."""
    p, lig, inj, nat, fasc, quality, fem, tib = scene["_ctx"]
    lay = scene.get("_lay")
    if lay is None or not lay.active:
        return []
    base_fail = {c["id"] for c in lay.conflicts if c["status"] == "fail"}
    out = []
    for c in lay.conflicts:
        if c["status"] != "fail":
            continue
        for fx in c["fixes"][:4]:
            if len(out) >= limit:
                return out
            p2 = dict(p)
            p2.update(fx["set"])
            errs, _ = validate_params(p2)
            if errs:
                out.append({"conflict": c["id"], "label": fx["label"], "set": fx["set"], "verified": False,
                            "result": "not applicable: " + errs[0]["message"]})
                continue
            if any(k in LIG_PARAMS for k in fx["set"]):
                out.append({"conflict": c["id"], "label": fx["label"], "set": fx["set"], "verified": None, "result": "changes the anatomy; not pre-tested"})
                continue
            l2 = LayeredScaffold(coerce_types(p2), lig, [fem, tib], "preview", inj, MATERIALS)
            fails = {x["id"] for x in l2.conflicts if x["status"] == "fail"}
            new = sorted(fails - base_fail)
            out.append({"conflict": c["id"], "label": fx["label"], "set": fx["set"], "verified": c["id"] not in fails,
                        "new_conflicts": new, "result": ("resolves it" if c["id"] not in fails else "does not resolve it") +
                        ("; creates: " + ", ".join(new) if new else "")})
    return out


def _runs(mask):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _chk(cid, label, status, value=None, threshold=None, units="", params=None, message="", category="geometry"):
    return {"id": cid, "label": label, "status": status, "value": value, "threshold": threshold, "units": units,
            "params": params or [], "message": message, "category": category}


def run_checks(p, prov, lig, inj, nat, fasc, scf, yarns, clear, objs, fem, tib, S, P, D, warns, quality="full", fitr=None):
    C = []
    fitr = fitr or {}
    for k, iss in enumerate(lig.issues):
        C.append(_chk("section_model_%d" % k, "Ligament geometry", iss["level"], None, None, "", [iss["param"]], iss["message"]))
    # footprints on bone
    for sd, bone in (("tib", tib), ("fem", fem)):
        g = lig.fp[sd].grid.reshape(-1, 3)
        dev = float(np.abs(bone.sdf(g)).max())
        C.append(_chk("footprint_on_bone_%s" % sd, "%s footprint lies on bone" % ("Tibial" if sd == "tib" else "Femoral"),
                      "pass" if dev < 0.25 else "fail", dev, 0.25, "mm", ["fp.%s.center" % sd],
                      "max distance of footprint points from the bone surface"))
    # mid-body penetration into bone
    s_lo = float(p["lig.flare_tib_mm"]) / lig.L
    s_hi = 1 - float(p["lig.flare_fem_mm"]) / lig.L
    mid = (S > s_lo) & (S < s_hi)
    depth = 0.0
    if mid.any():
        depth = float(max(0.0, -min(fem.sdf(P[mid]).min(), tib.sdf(P[mid]).min())))
    C.append(_chk("ligament_bone_penetration", "Ligament mid-body clears bone", "pass" if depth < 0.3 else ("warn" if depth < 1.0 else "fail"),
                  depth, 0.3, "mm", ["fp.fem.center", "fp.tib.center", "lig.bow_anterior_mm", "lig.bow_lateral_mm", "lig.width_profile", "lig.thickness_profile", "knee.flexion_deg"],
                  "deepest envelope point inside bone outside the insertion transitions"))
    # frame continuity
    dN = np.degrees(np.arccos(np.clip((lig.N1[1:] * lig.N1[:-1]).sum(1), -1, 1))).max()
    C.append(_chk("frame_continuity", "No frame flips along the centreline", "pass" if dN < 8 else "fail", float(dN), 8, "deg/station",
                  ["lig.ctrl_offsets", "lig.bow_anterior_mm"], "rotation-minimizing frames"))
    # fascicle packing
    lim = PACKING_LIMIT[p["native.packing_pattern"]]
    C.append(_chk("fascicle_packing", "Fascicle packing is achievable", "pass" if nat.packing <= lim else "fail",
                  nat.packing, lim, "", ["native.fascicle_count", "native.fascicle_diameter_mm", "native.fascicle_packing", "native.sizing_mode"],
                  "N*pi*d^2/4 / A_mid for the %s pattern" % p["native.packing_pattern"]))
    C.append(_chk("fascicle_overlap", "Fascicles do not overlap", "pass" if nat.min_gap_mm >= -0.03 else ("warn" if nat.min_gap_mm > -0.15 else "fail"),
                  nat.min_gap_mm, -0.03, "mm", ["native.fascicle_packing", "native.packing_pattern", "native.fascicle_count"],
                  "smallest surface gap between neighbouring fascicles at mid-length"))
    # meshes (audited on full builds; previews are followed by a full build)
    bad = []
    for o in (objs if quality == "full" else []):
        if o["kind"] != "mesh" or o["category"] in ("bone", "cartilage"):
            continue
        a = mesher.audit(o["V"], o["F"])
        if not a["closed"] or a["nonmanifold_edges"]:
            bad.append("%s (%d open, %d non-manifold)" % (o["id"], a["open_edges"], a["nonmanifold_edges"]))
    C.append(_chk("solid_meshes", "Tissue, injury and scaffold meshes are closed solids", "pass" if not bad else "warn",
                  len(bad), 0, "meshes", [], "; ".join(bad) or "all closed and edge-manifold"))
    # injury semantics
    if inj.active:
        ext = inj.region_extent()
        if inj.inv != "whole":
            unaffected = [fa for fa in fasc if not inj.in_region(fa["m"][len(fa["m"]) // 2:len(fa["m"]) // 2 + 1]).any()]
            broken = [fa["id"] for fa in unaffected if fa["damaged"].any()]
            C.append(_chk("partial_preserves", "Partial tear preserves unaffected tissue", "pass" if not broken else "fail",
                          len(broken), 0, "fascicles", ["injury.involvement", "injury.sector_center_deg", "injury.sector_width_deg"],
                          "%d unaffected fascicle groups remain continuous" % len(unaffected)))
        C.append(_chk("tear_within_ligament", "Tear lies within the ligament", "pass" if ext["lo_min"] > 0 and ext["hi_max"] < lig.L else "warn",
                      [ext["lo_min"], ext["hi_max"]], [0, lig.L], "mm", ["injury.center_s", "injury.gap_mm", "injury.retraction_tib_mm", "injury.retraction_fem_mm"],
                      "torn interval along the centreline"))
    # scaffold
    if scf.active and not yarns:
        C.append(_chk("scaffold_layout", "Scaffold yarns placed", "fail", 0, int(p["scaffold.yarn_count"]), "yarns",
                      ["scaffold.yarn_count", "scaffold.yarn_diameter_mm", "scaffold.fill_fraction", "scaffold.engagement"],
                      "; ".join(m["message"] for m in scf.messages) or "no yarn fits in the repair space"))
    if scf.active and yarns:
        F = fitr
        tol = F.get("tolerance_mm", 0.02)
        units = {u.id: u for u in scf.units}
        # each yarn must reach (or pass) its local tear face at both ends
        short = 0
        for y in yarns:
            if y["family"] == "core" and scf.engagement != "interdigitate":
                continue
            u = units[y["unit"]]
            gap = float(p["fit.end_gap_mm"])
            lim = gap + float(p["scaffold.clearance_mm"]) + 0.35          # end gap + sampling step
            if "end_gaps" in y:
                gt, gf = y["end_gaps"]
                reach_t = gt is None or gt <= lim
                reach_f = gf is None or gf <= lim
            else:
                lo, hi = inj.bounds(y["s"], y["m"], y["P"])
                reach_t = u.bone_tib or y["sig"][0] <= lo[0] + gap + 0.35 + float(y["r"][0])
                reach_f = u.bone_fem or y["sig"][-1] >= hi[-1] - gap - 0.35 - float(y["r"][-1])
            short += int(not (reach_t and reach_f))
        C.append(_chk("scaffold_spans_gap", "Scaffold spans the injury region", "pass" if short == 0 else "fail",
                      short, 0, "yarns short of a tear face", ["fit.end_gap_mm", "scaffold.overlap_tib_mm", "scaffold.overlap_fem_mm"],
                      "every abutting yarn end lies within the end gap of the remnant (its local tear face or the nearest retained fascicle surface)"))
        vol = F.get("volume_outside_envelope_mm3", 0.0)
        C.append(_chk("fit_envelope", "Scaffold stays inside the ligament envelope", "pass" if vol <= 1e-3 else "fail",
                      vol, 0.0, "mm3", ["fit.boundary_clearance_mm", "scaffold.yarn_diameter_mm", "scaffold.braid_radius_fraction", "scaffold.fill_fraction"],
                      "strand volume outside the intact envelope, measured in 3D on every yarn sample with its full radius "
                      "(tolerance %.2f mm; %.3f mm3 of %.1f mm3 strand volume)%s" % (
                          tol, vol, F.get("strand_volume_mm3", 0.0),
                          "; attachment interface at bone-anchored ends: %.3f mm3 (permitted)" % F["volume_in_attachment_interface_mm3"]
                          if F.get("volume_in_attachment_interface_mm3", 0) > 1e-3 else "")))
        vr = F.get("volume_outside_repair_space_mm3", 0.0)
        C.append(_chk("fit_repair_space", "Scaffold stays inside its repair region", "pass" if vr <= 1e-3 else "fail",
                      vr, 0.0, "mm3", ["fit.interface_clearance_mm", "fit.boundary_clearance_mm", "injury.involvement", "fit.partition"],
                      "strand volume outside the affected region of its section (contour, AM/PL interface, sector/mask edge), checked at every station"))
        if any(u.iface_side is not None for u in scf.units):
            nc = F.get("interface_crossings", 0)
            C.append(_chk("fit_bundle", "No scaffold crosses into the other bundle", "pass" if nc == 0 else "fail",
                          F.get("interface_crossing_depth_mm", 0.0), 0.0, "mm", ["fit.interface_clearance_mm", "bundle.am_fraction"],
                          "%d strand samples cross the AM/PL interface" % nc))
        if len(scf.units) > 1:
            uc = F.get("unit_collisions", 0)
            C.append(_chk("fit_units", "AM and PL repair units do not intersect", "pass" if not uc else "fail",
                          F.get("unit_min_clearance_mm"), 0.0, "mm", ["fit.interface_clearance_mm", "fit.partition"],
                          "smallest surface gap between strands of different repair units"))
        nb = F.get("samples_beyond_overlap", 0)
        C.append(_chk("fit_overlap", "No scaffold beyond the permitted overlap", "pass" if nb == 0 else "fail",
                      nb, 0, "samples", ["scaffold.overlap_tib_mm", "scaffold.overlap_fem_mm"],
                      "longitudinal extent stays within the damaged interval plus the permitted interface overlap"))
        mc = F.get("min_envelope_clearance_mm")
        want = float(p["fit.boundary_clearance_mm"])
        sh = F.get("clearance_shortfall_samples", 0)
        C.append(_chk("fit_clearance", "Requested clearances are met", "pass" if sh == 0 else "warn",
                      mc, want, "mm", ["fit.boundary_clearance_mm", "fit.interface_clearance_mm"],
                      "smallest distance from a strand surface to the ligament boundary (requested %.2f mm); %d samples closer than requested" % (want, sh)))
        bc = F.get("min_bone_clearance_mm")
        C.append(_chk("scaffold_bone_penetration", "Scaffold clears bone", "pass" if bc is None or bc >= -tol else "fail",
                      bc, 0.0, "mm", ["scaffold.bone_reach_mm", "fit.end_transition_mm"],
                      "smallest strand-surface distance to bone outside the permitted attachment interface"))
        if "yarn_min_clearance_mm" in clear:
            g = clear["yarn_min_clearance_mm"]
            C.append(_chk("yarn_clearance", "Scaffold yarns do not intersect", "pass" if g >= -0.01 else "fail",
                          g, 0.0, "mm", ["scaffold.yarn_count", "scaffold.yarn_diameter_mm", "scaffold.braid_angle_deg", "scaffold.clearance_mm"],
                          "%d samples closer than touching; min clearance %.3f mm (sampling tolerance 0.01 mm)" % (clear["yarn_collisions"], g)))
        if "tissue_min_clearance_mm" in clear:
            fr = clear["tissue_collision_fraction"]
            C.append(_chk("scaffold_tissue_clearance", "Scaffold avoids retained fascicles", "pass" if clear["tissue_collisions"] == 0 else ("warn" if fr < 0.005 else "fail"),
                          clear["tissue_min_clearance_mm"], 0.0, "mm", ["scaffold.engagement", "scaffold.yarn_count", "scaffold.overlap_tib_mm", "scaffold.overlap_fem_mm"],
                          "%.2f%% of yarn samples intersect retained fascicles (unintended overlap with retained tissue)" % (100 * fr)))
        for ur in F.get("units", []):
            if ur.get("braid_components") is not None:
                ok = ur["braid_components"] == 1
                C.append(_chk("fit_connectivity_%s" % ur["id"].lower(), "%s: braid is one connected structure" % ur["label"], "pass" if ok else "warn",
                              ur["braid_components"], 1, "components", ["scaffold.yarn_count", "scaffold.braid_angle_deg"],
                              "interlaced yarn groups after fitting; %d yarn pieces for %d yarns" % (ur["pieces"], ur["yarns"])))
            if ur["pieces"] != ur["yarns"]:
                C.append(_chk("fit_continuity_%s" % ur["id"].lower(), "%s: yarns are continuous" % ur["label"], "warn",
                              ur["pieces"], ur["yarns"], "pieces", [], "a yarn is split into several pieces"))
        for k, m in enumerate(scf.messages):
            C.append(_chk("scaffold_note_%d" % k, "Scaffold fit", m["level"], None, None, "", [m["param"]], m["message"]))
        has_bone = any(u.bone_tib or u.bone_fem for u in scf.units)
        C.append(_chk("bone_interface_rule", "Bone-interface phase only where the repair reaches bone", "pass", int(has_bone), None, "",
                      ["scaffold.bone_reach_mm"], "bone interface: %s" % (", ".join(sorted(set(
                          k for u in scf.units for k, v in (("tibial", u.bone_tib), ("femoral", u.bone_fem)) if v))) or "none (midsubstance repair)")))
    for w in warns:
        C.append(_chk("input_" + w["param"], "Input note", "warn", None, None, "", [w["param"]], w["message"], "input"))
    # provenance
    generic = [k for k, v in prov.items() if v.get("status") in ("default", "literature")]
    missing = [k for k, v in prov.items() if v.get("status") in ("missing", "awaiting_review")]
    C.append(_chk("provenance", "Patient-specific inputs", "warn" if (missing or len(generic) > 0) else "pass", len(missing), 0, "",
                  missing[:20], "%d parameters use generic defaults or literature values; %d imported values are missing or awaiting review"
                  % (len(generic), len(missing)), "provenance"))
    for cid, lab in (("mechanics", "Strength / stiffness"), ("degradation", "Degradation"), ("healing", "Healing / integration"),
                     ("retear_risk", "Re-tear risk")):
        C.append(_chk(cid, lab, "not_evaluated", None, None, "", [], "not modelled in this application", "scope"))
    return C


def collect_metrics(p, lig, inj, nat, scf, ssum, clear, remnant_info):
    m = {"ligament_length_mm": lig.L, "inclinations": lig.inclinations,
         "inclination_reference": {"sagittal_deg": 47.4, "coronal_deg": 75.3, "source": "L03"},
         "footprint_area_mm2": {"tibial": lig.fp["tib"].area(), "femoral": lig.fp["fem"].area()},
         "midsubstance_area_mm2": float(lig.section_area(0.5)[0]),
         "fascicles": {"count": nat.n_fasc, "diameter_mm": nat.d_fasc, "packing": nat.packing,
                       "per_bundle": [b["count"] for b in nat.bundle_info], "min_gap_mm": nat.min_gap_mm,
                       "bundle_twist_total_deg": [math.degrees(g) for g in nat.gamma_tot],
                       "sizing_mode": p["native.sizing_mode"]},
         "fibers": nat.physical_counts(),
         "crimp_angle_deg": {"native": math.degrees(math.atan(2 * math.pi * float(p["native.crimp_amplitude_um"]) / float(p["native.crimp_wavelength_um"]))),
                             "scaffold": math.degrees(math.atan(2 * math.pi * float(p["scaffold.crimp_amplitude_um"]) / float(p["scaffold.crimp_wavelength_um"])))},
         "bundle_rotation_total_deg": float(p["bundle.rotation_total_deg"]),
         "footprint_alignment_deg": {k: math.degrees(v) for k, v in lig.align.items()},
         "length": lig.length_info, "section_profile": lig.section_report(),
         "bundles": [{"name": BUNDLE_NAMES[b], "fascicles": bi["count"], "requested": bi["requested"],
                      "fascicle_diameter_mm": bi["diameter_mm"], "packing": bi["packing"], "area_mid_mm2": bi["area_mid_mm2"],
                      "within_bundle_rotation_deg": math.degrees(nat.gamma_tot[b]),
                      "outermost_fascicle_angle_deg": nat.fascicle_angle_deg[b]} for b, bi in enumerate(nat.bundle_info)],
         "structure_mode": p["bundle.structure_mode"], "section_mode": p["lig.section_mode"]}
    if inj.active:
        ext = inj.region_extent()
        m["injury_bundles"] = None
        if inj.pb is not None:
            m["injury_bundles"] = [{"bundle": BUNDLE_NAMES[k], "center_s": inj.pb[k]["sig0"] / lig.L, "gap_mm": 2 * inj.pb[k]["half"],
                                    "depth_fraction": inj.pb[k]["depth"],
                                    "affected_area_fraction_of_bundle": float(inj.in_region(sunflower(3000))[lig.bundle_of(sunflower(3000)) == k].mean())}
                                   for k in (0, 1)]
        m["injury"] = {"type": inj.type, "involvement": inj.inv, "tibial_stump_min_mm": ext["lo_min"],
                       "femoral_stump_min_mm": lig.L - ext["hi_max"], "nominal_gap_mm": float(p["injury.gap_mm"]) + float(p["injury.retraction_tib_mm"]) + float(p["injury.retraction_fem_mm"]),
                       "affected_area_fraction": ext["area_fraction"], "boundary_source": "observed mask" if inj.mask is not None else ("observed table" if inj.table else "plane model")}
        stumps = [v for oid, v in remnant_info if "stump" in oid]
        if len(stumps) >= 2:
            d, _ = cKDTree(stumps[0]).query(stumps[1])
            m["injury"]["measured_gap_mm"] = float(d.min())
    if scf.active:
        m["scaffold"] = ssum
        m["scaffold"]["clearance"] = {k: v for k, v in clear.items() if k != "yarn_bad_ids"}
    return m


ROT_PARAMS = {"native bundle": ["native.fascicle_rotation_deg", "am.fascicle_rotation_deg", "pl.fascicle_rotation_deg"],
              "native fascicle": ["native.fiber_rotation_deg", "am.fiber_rotation_deg", "pl.fiber_rotation_deg"],
              "scaffold repair unit": ["scaffold.yarn_rotation_deg"], "scaffold yarn": ["scaffold.fiber_rotation_deg"],
              "braided repair unit": ["scaffold.arch", "scaffold.braid_angle_deg", "scaffold.braid_pitch_mm"]}


def rotation_checks(rotr):
    """Surgeon-informed rule: fibers within each cylindrical unit rotate at most 30 deg in
    total (signed), measured on the generated paths."""
    out = []
    lim, tol = rotr["limit_deg"], rotr["tolerance_deg"]
    for k, r in enumerate(rotr["rows"]):
        if not r.get("n"):
            continue
        lvl = r["level"].split(":")[0]
        params = next((v for kk, v in ROT_PARAMS.items() if lvl.startswith(kk)), [])
        mx = r["net_max_abs_deg"]
        if r.get("braid_winding"):
            out.append(_chk("rotation_braid_%d" % k, "%s: braid yarns wind around the unit axis" % r["unit"], "warn" if mx > lim + tol else "pass",
                            mx, lim, "deg", params,
                            "braid yarns wind up to %.0f deg (end to end) around the repair unit's axis; a braid needs this winding for its crossings. "
                            "The surgeon-informed +-30 deg rule is enforced for the fibers inside each yarn; if it must also hold for the braid's "
                            "yarns, use aligned yarns or a braid whose total winding is <= 30 deg" % mx, "rotation"))
            continue
        req = r["requested_deg"]
        status = "pass"
        msg = "measured on the generated paths of %d unit(s) (%d paths): largest unit total %+.1f deg (requested %+.1f deg); " \
              "back-and-forth <= %.1f deg; single paths deviate up to %.1f deg (dispersion, crimp, section shape)" % (
                  r["n"], r.get("paths", 0), r["net_at_max_deg"], req, r["excursion_max_deg"], r.get("path_spread_deg", 0.0))
        if "region_turn_deg" in r:
            msg += "; the bundle region itself turns %+.1f deg with the ligament's shape (anatomy, not counted)" % r["region_turn_deg"]
        act = r.get("actual_path_max_abs_deg")
        if act is not None:
            msg += "; actual paths turn at most %.1f deg (yarns trimmed at the tear faces turn by their share of the unit)" % act
            mx = max(mx, act) if act > lim + tol else mx
        if mx > lim + tol:
            status = "fail"
            msg = "exceeds the surgeon-informed range of +-%g deg: " % lim + msg
        elif abs(r["net_at_max_deg"] - req) > max(3.0, 0.15 * abs(req)) or r["excursion_max_deg"] > 3.0:
            status = "warn"
            msg = "generated rotation differs from the requested value or is not monotone: " + msg
        out.append(_chk("rotation_%d" % k, "Total fiber rotation: %s (%s)" % (r["level"], r["unit"]), status, r["net_at_max_deg"], req, "deg",
                        params, msg, "rotation"))
    return out


def _section_pos(disp):
    kind = disp.get("section_kind", "ligament")
    if kind in ("ligament", "am", "pl"):
        return float(disp.get("section_s", 0.5))
    return float(disp.get("section_offset_mm", 0.0))


def _scaffold_ok(p, lig, inj, nat, fasc, quality, fem, tib):
    """Scaffold-only rebuild + the hard fit criteria (used to verify resolutions)."""
    scf = Scaffold(p, lig, inj, nat, fasc, quality, bones=[fem, tib])
    if not scf.active:
        return False
    ys = scf.yarns([fem, tib])
    if not ys or any(m["level"] == "fail" for m in scf.messages):
        return False
    clear = scf.clearance(ys, fasc)
    if clear.get("yarn_min_clearance_mm", 0) < -0.01 or clear.get("tissue_collision_fraction", 0) >= 0.005:
        return False
    F = fitmod.evaluate(lig, inj, scf, ys, clear, [fem, tib], fasc)
    return (F.get("volume_outside_envelope_mm3", 0) <= 1e-3 and F.get("volume_outside_repair_space_mm3", 0) <= 1e-3
            and not F.get("interface_crossings") and not F.get("samples_beyond_overlap") and not F.get("unit_collisions")
            and (F.get("min_bone_clearance_mm") is None or F["min_bone_clearance_mm"] >= -0.02))


def _generic_candidates(p, scf, yarns, fails):
    """Fallback fixes for a scaffold that does not fit (each one is verified before
    it is offered)."""
    ids = {c["id"] for c in fails}
    out = []
    d0 = float(p["scaffold.yarn_diameter_mm"])
    if ids & {"scaffold_layout", "yarn_clearance"} or any(i.startswith("scaffold_note") for i in ids):
        for f in (0.8, 0.65, 0.5):
            d = math.floor(d0 * f * 100) / 100
            if d >= 0.1:
                out.append({"param": "scaffold.yarn_diameter_mm", "value": d, "label": "Use %.2f mm yarns (base)" % d})
        if p["scaffold.arch"] == "braided":
            n = int(p["scaffold.yarn_count"])
            for m in (n - 4, n - 8):
                if m >= 6:
                    out.append({"param": "scaffold.yarn_count", "value": m, "label": "Use %d carriers" % m})
            out.append({"param": "scaffold.arch", "value": "aligned", "label": "Use aligned yarns instead of a braid"})
    return out


def verify_resolutions(scene, limit=8):
    """Try each candidate fix (scaffold rebuilt with that one change) and return
    only the ones that make the scaffold fit. Nothing is applied."""
    ctx = scene.get("_ctx")
    if not ctx or not scene.get("resolution_candidates"):
        return []
    p, lig, inj, nat, fasc, quality, fem, tib = ctx
    return _verify_resolutions(p, lig, inj, nat, fasc, quality, fem, tib, scene["resolution_candidates"], limit)


def _verify_resolutions(p, lig, inj, nat, fasc, quality, fem, tib, res, limit=8):
    out = []
    done = set()
    tried = 0
    for r in res:
        if r["param"] in done:
            continue                        # first (least drastic) working value per parameter
        if tried >= limit:
            break
        tried += 1
        p2 = dict(p)
        p2[r["param"]] = r["value"]
        if validate_params(p2)[0]:
            continue
        try:
            ok = _scaffold_ok(p2, lig, inj, nat, fasc, quality, fem, tib)
        except Exception:
            ok = False
        if ok:
            out.append(dict(r, verified=True))
            done.add(r["param"])
    return out


def _dedupe(res):
    seen, out = set(), []
    for r in res:
        k = (r["param"], json_key(r["value"]))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def json_key(v):
    return repr(v)


def _unit_ring(lig, u):
    """Material outline of a unit's region, or None when it is not a simple ring
    (sector / mask / depth regions are meshed from their fields instead)."""
    if u.restricted:
        return None
    if u.iface_side is not None:
        return secmod.bundle_ring(lig, u.iface_side)
    ph = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    return np.stack([np.cos(ph), np.sin(ph)], 1)


def _unit_volume(lig, u, ring, a, b, h):
    """Closed mesh of a unit's region between arc positions a and b (mm)."""
    L = lig.L
    if ring is not None:
        return secmod.ring_loft(lig, ring, a / L, b / L, max(6, int((b - a) / 0.25) + 1))
    pts = []
    for f in u.fields:
        if a - 1e-6 <= f.sig <= b + 1e-6:
            G = f.grid()
            sel = f.Danat.ravel() >= 0
            Gs = G[sel][::4]
            if Gs.shape[0]:
                pts.append(lig.frame_point(np.full(Gs.shape[0], f.s), Gs))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return mesher.mesh_from_points(np.vstack(pts), h=h * 1.2, smooth=3)


def _markers(points, r):
    """Small octahedra at the given points (one mesh)."""
    base = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * r
    tri = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    V, F = [], []
    for k, P in enumerate(points):
        V.append(base + np.asarray(P, float))
        F.append(tri + 6 * k)
    return np.vstack(V), np.vstack(F).astype(np.int64)


def detail_fibers(state, sel, quality="full"):
    """High-resolution fibers of one fascicle or yarn at PHYSICAL crimp scale near s."""
    params = dict(state["params"])
    migrate_params(params)
    p = coerce_types(params)
    flex = float(p["knee.flexion_deg"])
    assets = state.get("assets") or {}
    fem = _load_bone(assets.get("femur_mesh"), "femur", flex) or ProceduralBone("femur", p, flex)
    tib = _load_bone(assets.get("tibia_mesh"), "tibia", 0.0) or ProceduralBone("tibia", p, 0.0)
    lig = Ligament(p, fem, tib, quality)
    inj = Injury(p, lig, assets)
    disp = dict(DISPLAY_DEFAULTS, **(state.get("display") or {}))
    nat = NativeTissue(p, lig, inj, disp)
    fasc = nat.fascicles()
    s0 = float(sel.get("s", disp["section_s"]))
    win = float(sel.get("window_mm", 1.5)) / lig.L
    n_rep = int(sel.get("fibers", 19))
    lines = []
    if sel.get("type") == "yarn":
        scf = Scaffold(p, lig, inj, nat, fasc, quality, bones=[fem, tib])
        if not scf.active:
            return {"ok": False, "error": "no scaffold in this design"}
        yarns = scf.yarns([fem, tib])
        y = [y for y in yarns if y["id"] == int(sel["id"])]
        if not y:
            return {"ok": False, "error": "yarn not found"}
        sig = s0 * lig.L
        fr = scf.fibers(y, n_rep, 1.0, (sig - win * lig.L, sig + win * lig.L))
        for f in fr:
            for k in range(f["P"].shape[0]):
                lines.append({"P": f["P"][k], "r": np.full(f["P"].shape[1], f["r"][k]), "fiber": k, "region": f["region"]})
        parent = {"P": y[0]["P"], "r": y[0]["r"]}
        note = "scaffold yarn %d: total fiber rotation along the yarn %+.1f deg" % (int(sel["id"]), float(p["scaffold.fiber_rotation_deg"]))
    else:
        k = int(sel.get("id", 0))
        fa = [f for f in fasc if f["id"] == k]
        if not fa:
            return {"ok": False, "error": "fascicle not found"}
        fr = nat.fibers(fa, n_rep, 1.0, (max(0.0, s0 - win), min(1.0, s0 + win)))
        for f in fr:
            for j in range(f["P"].shape[0]):
                lines.append({"P": f["P"][j], "r": np.full(f["P"].shape[1], f["r"][j]), "fiber": j})
        parent = {"P": fa[0]["P"], "r": fa[0]["r"]}
        note = ("fascicle %d (%s): crimp %.0f um / %.0f um at true scale" %
                (k, BUNDLE_NAMES[fa[0]["bundle"]], float(p["native.crimp_amplitude_um"]), float(p["native.crimp_wavelength_um"])))
    if state.get("params", {}).get("case.side") == "left":
        for ln in lines:
            ln["P"] = ln["P"] * np.array([-1, 1, 1])
    return {"ok": True, "lines": lines, "note": note, "window_mm": 2 * win * lig.L,
            "physical": {"native_fiber_diameter_um": float(p["native.fiber_diameter_um"]),
                         "scaffold_fiber_diameter_um": float(p["scaffold.fiber_diameter_um"])}}
