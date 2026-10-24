"""End-to-end bilateral case pipeline.

QC -> bilateral registration -> anatomy estimates (+ separated uncertainty) ->
abstention -> [scaffold candidate -> validation -> export gate -> export] ->
[simplified mechanics] -> research report.

Every intermediate is written to disk with its evidence level so the interface and
reports can show what is observed, estimated, hypothesised or synthetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from . import abstention
from .annotation.schema import AnnotationRecord, Label, read_slicer_markups
from .estimation.anatomy import estimate_contralateral, reconcile_with_remnants, uncertainty_ensembles, uncertainty_summary
from .imaging.preprocess import qc_png
from .imaging.volume import Volume, load_nifti, save_nifti
from .provenance import EvidenceLevel, Provenance, ReviewStatus, sha256_file, utcnow, write_json
from .registration import bilateral as B


def load_review(run_dir: Path) -> dict:
    p = run_dir / "review.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"anatomy_estimate": {"status": ReviewStatus.UNREVIEWED.value, "reviewer": None, "selected_estimate": None, "history": []}}


def set_review(run_dir: Path, status: str, reviewer: str, selected: str | None = None, note: str = "") -> dict:
    r = load_review(run_dir)
    a = r["anatomy_estimate"]
    a["history"].append({"time": utcnow(), "status": status, "reviewer": reviewer, "selected": selected, "note": note})
    a.update(status=ReviewStatus(status).value, reviewer=reviewer, selected_estimate=selected or a.get("selected_estimate"))
    write_json(run_dir / "review.json", r)
    return r


def run_case(case_dir, out_dir, scaffold_cfg: dict | None = None, mechanics_cfg: dict | None = None, ensemble_size: int = 8,
             seed: int = 0, thresholds: dict | None = None, log=print) -> dict:
    case_dir, out = Path(case_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    case = json.loads((case_dir / "case.json").read_text())
    synthetic = bool(case.get("synthetic"))
    res: dict = {"case_id": case["case_id"], "synthetic": synthetic, "started": utcnow(),
                 "evidence_banner": "SYNTHETIC DEMONSTRATION - not patient data" if synthetic else ""}
    vol = {}
    for role in ("injured", "contralateral"):
        vol[f"{role}_image"] = load_nifti(case_dir / case[role]["image"])
        vol[f"{role}_labels"] = load_nifti(case_dir / case[role]["labels"])
        ap = case_dir / str(case[role]["labels"]).replace(".nii.gz", ".annotation.json")
        vol[f"{role}_annotation"] = AnnotationRecord.load(ap) if ap.exists() else None
        lp = case_dir / case[role].get("landmarks", "missing")
        vol[f"{role}_landmarks"] = read_slicer_markups(lp) if lp.exists() else {}
        qc_png(vol[f"{role}_image"], out / "qc" / f"{role}.png", label=vol[f"{role}_labels"].data, title=f"{case['case_id']} {role} ({case[role]['side']})")
        res[f"{role}_image"] = vol[f"{role}_image"].summary()
        res[f"{role}_inputs_sha256"] = sha256_file(case_dir / case[role]["image"])
    il, cl = vol["injured_labels"], vol["contralateral_labels"]
    log("registering (mirror + per-bone rigid ICP)")
    pre = abstention.evaluate(case, vol["injured_image"], vol["contralateral_image"], None, None, None, None, None, synthetic, thresholds)
    if pre["abstain_estimation"]:
        res["abstention"] = pre
        res["status"] = "abstained before estimation"
        write_json(out / "pipeline.json", res)
        return res
    con_lm = vol["contralateral_landmarks"]
    con_fp = None
    if "acl_femoral_footprint_center" in con_lm and "acl_tibial_footprint_center" in con_lm:
        con_fp = {"femoral": np.array(con_lm["acl_femoral_footprint_center"]), "tibial": np.array(con_lm["acl_tibial_footprint_center"])}
    reg = B.register_bilateral(il, cl, seed=seed, con_footprints=con_fp)
    qc = B.registration_qc(reg, il, cl, vol["injured_landmarks"], con_lm)
    write_json(out / "registration.json", {**reg.to_dict(), "qc": qc, "evidence_level": EvidenceLevel.ESTIMATED.value})
    res["registration"] = {**reg.to_dict(), "qc": qc}
    # mirrored + registered contralateral image on the injured grid (inspection only)
    mir = B.transfer_to_injured(vol["contralateral_image"], reg, vol["injured_image"], kind="image")
    save_nifti(vol["injured_image"].with_data(mir), out / "inspection" / "contralateral_mirrored_registered.nii.gz", np.float32)
    mir_lab = B.transfer_to_injured(cl, reg, il, kind="label")
    save_nifti(il.with_data(mir_lab.astype(np.uint8)), out / "inspection" / "contralateral_labels_mirrored_registered.nii.gz", np.uint8)
    log("estimating native ACL geometry")
    a = estimate_contralateral(il, cl, reg)
    inj_lm = {k: v for k, v in vol["injured_landmarks"].items() if k.startswith("acl_")}
    b = reconcile_with_remnants(a, il, inj_lm)
    ests = {"contralateral_mirror": a, "remnant_reconciled": b}
    for name, e in ests.items():
        save_nifti(il.with_data(e.mask.astype(np.uint8)), out / "estimates" / f"{name}.nii.gz", np.uint8)
    res["estimates"] = {n: e.summary(il.affine) for n, e in ests.items()}
    log(f"uncertainty ensembles (n={ensemble_size} per source)")
    ens = uncertainty_ensembles(il, cl, reg, {"ensemble_size": ensemble_size}, seed=seed)
    for src, u in ens.items():
        save_nifti(il.with_data(u["probability"].astype(np.float32)), out / "estimates" / f"probability_{src}.nii.gz", np.float32)
    unc = uncertainty_summary(ens)
    unc["not_represented"] = ["pre-injury shape variation beyond the asymmetry prior", "fibre-level microstructure (not observable on routine MRI)",
                              "annotation error of the injured-side remnants"]
    write_json(out / "uncertainty.json", unc)
    res["uncertainty"] = unc
    abst = abstention.evaluate(case, vol["injured_image"], vol["contralateral_image"], reg.to_dict(), qc, b.conflicts, b.metrics,
                               {k: v for k, v in unc.items() if isinstance(v, dict)}, synthetic, thresholds)
    write_json(out / "abstention.json", abst)
    res["abstention"] = abst
    review = load_review(out)
    res["review"] = review
    if scaffold_cfg:
        res["scaffold"] = _scaffold(out, case, vol, ests, unc, abst, review, scaffold_cfg, synthetic, log, mechanics_cfg)
    res["finished"] = utcnow()
    res["provenance"] = Provenance(EvidenceLevel.SYNTHETIC if synthetic else EvidenceLevel.ESTIMATED, "bilacl.pipeline.run_case",
                                   inputs=[str(case_dir)], parameters={"seed": seed, "ensemble_size": ensemble_size, "scaffold_cfg": scaffold_cfg},
                                   review_status=ReviewStatus(review["anatomy_estimate"]["status"])).to_dict()
    res["status"] = "complete"
    write_json(out / "pipeline.json", res)
    from .report import write_report

    write_report(out, res)
    return res


def _scaffold(out, case, vol, ests, unc, abst, review, cfg, synthetic, log, mechanics_cfg):
    from .mechanics.axial import AxialModelInputs, axial_response, sensitivity, solid_area_profile
    from .scaffold.design import ScaffoldParams, generate_candidate
    from .scaffold.mesh import export_candidate, occupancy_to_mesh, validate_candidate

    il = vol["injured_labels"]
    which = review["anatomy_estimate"].get("selected_estimate") or cfg.get("estimate", "remnant_reconciled")
    est = ests[which]
    pdict = {k: v for k, v in cfg.items() if k in ScaffoldParams.__dataclass_fields__}
    params = ScaffoldParams(**pdict)
    log(f"generating scaffold candidate from '{which}' ({params.mode}, pattern={params.alignment_pattern})")
    cand = generate_candidate(est.mask, est.femoral_fp, est.tibial_fp, il.data, il.affine, params)
    sres = {"estimate_used": which, "notes": cand.notes, "applicable": cand.applicable}
    if not cand.applicable:
        write_json(out / "scaffold" / "not_applicable.json", sres)
        return sres
    mesh = occupancy_to_mesh(cand.occupancy, cand.grid_affine)
    val = validate_candidate(cand, mesh, il.data, il.affine)
    sres["validation"] = val
    rv = ReviewStatus(review["anatomy_estimate"]["status"])
    gate = abstention.export_gate(abst, rv, val["geometric_checks_passed"], synthetic, vol["injured_image"].physical.verified)
    sres["export_gate"] = gate
    assumptions = [f"{k}: {v}" for k, v in params.assumption_sources.items()]
    assumptions += ["envelope = per-station ellipse fit to the selected anatomy estimate (estimated, not observed)",
                    "internal channels follow a user-selected alignment hypothesis; not derived from imaged fibres"]
    if gate["allowed"]:
        source = {"case_id": case["case_id"], "injured_image": case["injured"]["image"], "injured_side": case["injured"]["side"],
                  "image_affine_ras": il.affine.tolist(), "source_spacing_mm": vol["injured_image"].spacing.tolist(),
                  "registration": json.loads((out / "registration.json").read_text())}
        prov = Provenance(EvidenceLevel.SYNTHETIC if synthetic else EvidenceLevel.HYPOTHESIZED, "bilacl.scaffold", inputs=[case["case_id"]],
                          parameters=pdict, review_status=rv, reviewer=review["anatomy_estimate"].get("reviewer"))
        sres["files"] = export_candidate(cand, mesh, val, out / "scaffold", source=source, gate=gate, provenance=prov,
                                         uncertainty={k: v for k, v in unc.items()}, assumptions=assumptions, synthetic=synthetic)
        log(f"exported {sres['files']['stl']}")
    else:
        write_json(out / "scaffold" / "export_blocked.json", {"gate": gate, "validation": val, "notes": cand.notes})
        log("export blocked: " + "; ".join(gate["reasons"]))
    if mechanics_cfg:
        prof = solid_area_profile(cand.occupancy, cand.grid_affine, est.femoral_fp, est.tibial_fp)
        inp = AxialModelInputs(**{k: v for k, v in mechanics_cfg.items() if k in AxialModelInputs.__dataclass_fields__})
        try:
            mech = axial_response(prof, inp)
            if mechanics_cfg.get("sensitivity"):
                mech["sensitivity"] = sensitivity(prof, inp, mechanics_cfg["sensitivity"])
            mech["profile"] = {"s_mm": prof["s_mm"].tolist(), "area_mm2": prof["area_mm2"].tolist()}
            mech["label"] = "SIMPLIFIED 1D MODEL - not finite-element; no stress field" + (" - SYNTHETIC" if synthetic else "")
        except ValueError as e:
            mech = {"status": "blocked", "reason": str(e)}
        write_json(out / "scaffold" / "mechanics.json", mech)
        sres["mechanics"] = mech
    return sres


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text()) if path else {}
