"""bilacl command-line interface. Run `bilacl -h` or `bilacl <command> -h`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _p(obj):
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------- commands


def cmd_hardware(a):
    import os
    import platform
    import shutil

    import torch

    _p({"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "torch": torch.__version__, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
        "disk_free_gb": round(shutil.disk_usage(".").free / 1e9, 1)})


def cmd_registry(a):
    from .acquisition import load_registry

    reg = load_registry()
    if a.markdown:
        for k, d in reg.items():
            print(f"## {d.get('name')} (`{k}`)\nsource: {d.get('official_source')}  verified_on: {d.get('verified_on')}\n")
            for f, v in d.items():
                if isinstance(v, dict) and "state" in v:
                    print(f"- **{f}** [{v['state']}]: {v['value']}")
            print()
    else:
        _p(reg)


def cmd_acquire(a):
    from .acquisition import acquisition_report, download, load_registry

    if a.action == "report":
        rep = acquisition_report(a.data_root, probe_remote=not a.no_remote)
        if a.out:
            Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
        _p(rep)
    else:
        reg = load_registry()[a.dataset]
        if reg.get("download_mode") != "public_http":
            sys.exit(f"{a.dataset}: automated download not permitted ({reg.get('download_mode')}). "
                     f"Obtain it from {reg.get('official_source')} under its terms and place it in {a.data_root}/{a.dataset}.")
        url = reg["download_url"]
        _p(download(url, Path(a.data_root) / a.dataset / Path(url).name, confirm=a.confirm, sha256=a.sha256, allow_large=a.allow_large))


def cmd_inspect(a):
    from .adapters.base import get_adapter

    _p(get_adapter(a.adapter, a.root).inspect().__dict__)


def cmd_manifest(a):
    from .acquisition import select_dev_subset
    from .adapters.base import get_adapter
    from .manifest import read_manifest, summarize, write_manifest
    from .splits import assign_splits, audit_leakage

    if a.action == "build":
        recs = []
        for spec in a.sources:
            name, root = spec.split("=", 1)
            recs += get_adapter(name, root).records()
        write_manifest(recs, a.out)
        _p(summarize(recs))
    elif a.action == "split":
        recs = read_manifest(a.manifest)
        info = assign_splits(recs, seed=a.seed)
        write_manifest(recs, a.out or a.manifest)
        _p({"assignment": info, "leakage": audit_leakage(recs), "summary": summarize(recs)})
    elif a.action == "audit":
        _p(audit_leakage(read_manifest(a.manifest)))
    elif a.action == "summary":
        _p(summarize(read_manifest(a.manifest)))
    elif a.action == "subset":
        recs = select_dev_subset(read_manifest(a.manifest), a.n_subjects, a.seed)
        write_manifest(recs, a.out)
        _p(summarize(recs))


def cmd_dicom(a):
    from .imaging.volume import discover_dicom_series, load_dicom_series, save_nifti, select_series
    from .provenance import write_json

    ranked = select_series(discover_dicom_series(a.dir), prefer=a.prefer)
    rep = [{"series_uid": r.series_uid, "description": r.description, "usable": r.usable, "localizer": r.is_localizer,
            "n_files": r.n_files, "n_slices": r.n_slices_used, "laterality": r.laterality, "laterality_source": r.laterality_source,
            "issues": r.issues, "warnings": r.warnings, "meta": {k: v for k, v in r.meta.items() if not k.startswith("_")}} for r, _ in ranked]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "series_report.json", rep)
    if a.convert:
        for r, items in ranked:
            if r.usable:
                v = load_dicom_series(items, r)
                save_nifti(v, out / f"{r.series_uid[-16:]}.nii.gz")
                write_json(out / f"{r.series_uid[-16:]}.json", v.meta)
    _p(rep)


def cmd_preprocess(a):
    from .imaging.preprocess import intensity_normalize, n4_bias_correct, qc_png, resample_spacing
    from .imaging.volume import load_any, save_nifti
    from .provenance import write_json

    out = Path(a.out)
    v = load_any(a.image)
    lab = load_any(a.labels) if a.labels else None
    qc_png(v, out / "qc_before.png", label=lab.data if lab else None, title="source")
    if a.n4:
        v = n4_bias_correct(v)
    if a.spacing:
        v = resample_spacing(v, a.spacing, "image", order=a.order)
        if lab:
            lab = resample_spacing(lab, a.spacing, "label")
    v = intensity_normalize(v)
    save_nifti(v, out / "image.nii.gz")
    if lab:
        save_nifti(lab, out / "labels.nii.gz")
    qc_png(v, out / "qc_after.png", label=lab.data if lab else None, title="preprocessed")
    write_json(out / "preprocess.json", {"summary": v.summary(), "meta": v.meta})
    _p(v.summary())


def cmd_annotate(a):
    import numpy as np

    from .annotation import schema as S
    from .imaging.volume import load_any, save_nifti

    if a.action == "color-table":
        S.write_slicer_color_table(a.out)
    elif a.action == "export-slicer":
        v = load_any(a.labels)
        S.write_slicer_seg_nrrd(v.data, v.affine, a.out)
    elif a.action == "import-slicer":
        lab, aff, info = S.read_slicer_seg_nrrd(a.seg)
        ref = load_any(a.reference)
        from .imaging.preprocess import resample_like
        from .imaging.volume import PhysicalStatus, Volume

        v = resample_like(Volume(lab, aff, PhysicalStatus(True, "slicer")), ref, kind="label")
        save_nifti(v, a.out, np.uint8)
        rec = S.AnnotationRecord(annotation_id=Path(a.out).name, exam_id=a.exam_id, image_ref=a.reference, annotator=a.annotator,
                                 annotator_role=a.role, version=a.version, parent_annotation_id=a.parent,
                                 annotated_labels=sorted({int(x) for x in np.unique(v.data)} - {0, 255}) if not a.annotated else a.annotated,
                                 source=a.source, model_id=a.model_id, tear_location=a.tear_location)
        errs = rec.validate()
        if errs:
            sys.exit("annotation record invalid: " + "; ".join(errs))
        rec.save(str(a.out).replace(".nii.gz", ".annotation.json"))
        _p({"segments": info, "record": rec.to_dict()})
    elif a.action == "agreement":
        masks = {Path(p).name: load_any(p).data for p in a.masks}
        _p(S.inter_reader_agreement(masks))
    elif a.action == "review":
        rec = S.AnnotationRecord.load(a.record)
        from .provenance import ReviewStatus

        rec.review_status = ReviewStatus(a.status)
        rec.reviewer = a.reviewer
        rec.save(a.record)
        _p(rec.to_dict())


def cmd_synth(a):
    from .synthetic.phantom import make_bilateral_case

    paths = []
    for i in range(a.n):
        s = a.seed + i
        paths.append(str(make_bilateral_case(Path(a.out) / f"SYN{s:04d}_{a.injury}", seed=s, injury=a.injury, injured_side=a.side)))
    _p({"cases": paths, "WARNING": "synthetic phantoms - no clinical meaning"})


def _records_for(manifest, task, allow_synthetic):
    from .manifest import read_manifest
    from .training.tasks import eligibility

    recs = read_manifest(manifest)
    el = eligibility(task, recs, allow_synthetic=allow_synthetic)
    return el


def cmd_tasks(a):
    from .manifest import read_manifest
    from .training.tasks import blocked_task_report

    _p(blocked_task_report(read_manifest(a.manifest)))


def cmd_train(a):
    from .training.seg import TrainConfig, train

    cfg = TrainConfig.from_yaml(a.config)
    if a.out:
        cfg.out_dir = a.out
    el = _records_for(a.manifest, cfg.task, cfg.allow_synthetic)
    tr = [r for r in el["eligible"] if r.split == "train"]
    va = [r for r in el["eligible"] if r.split == "val"]
    print(f"task {cfg.task}: {len(tr)} train / {len(va)} val eligible; {el['n_excluded']} excluded")
    if len(tr) < 1:
        _p({"status": "BLOCKED", "reason": "no eligible training records", "exclusions": el["excluded"][:20]})
        sys.exit(2)
    if any(r.split == "test" for r in tr + va):
        sys.exit("test records must not be used for training/model selection")
    run = train(cfg, tr, va)
    _p({k: run[k] for k in ("model_status", "best_val_mean_dice", "device", "note") if k in run})


def cmd_infer(a):
    from .training.seg import infer

    _p(infer(a.checkpoint, a.image, a.out))


def cmd_register(a):
    from .annotation.schema import read_slicer_markups
    from .imaging.volume import load_nifti
    from .provenance import write_json
    from .registration.bilateral import register_bilateral, registration_qc

    c = json.loads((Path(a.case) / "case.json").read_text())
    il = load_nifti(Path(a.case) / c["injured"]["labels"])
    cl = load_nifti(Path(a.case) / c["contralateral"]["labels"])
    reg = register_bilateral(il, cl)
    lm = lambda r: read_slicer_markups(Path(a.case) / c[r]["landmarks"]) if (Path(a.case) / c[r].get("landmarks", "x")).exists() else {}  # noqa: E731
    qc = registration_qc(reg, il, cl, lm("injured"), lm("contralateral"))
    d = {**reg.to_dict(), "qc": qc}
    write_json(Path(a.out) / "registration.json", d)
    _p(d)


def cmd_run(a):
    from .pipeline import load_config, run_case

    r = run_case(a.case, a.out, load_config(a.scaffold_config) if a.scaffold_config else None,
                 load_config(a.mechanics_config) if a.mechanics_config else None, ensemble_size=a.ensemble, seed=a.seed)
    print(f"status: {r['status']}; report: {Path(a.out) / 'report.md'}")


def cmd_review(a):
    from .pipeline import set_review

    _p(set_review(Path(a.run), a.status, a.reviewer, a.select, a.note))


def cmd_experiment(a):
    import numpy as np

    from .estimation.experiment import healthy_bilateral_holdout, markdown_table, synthetic_injury_experiment
    from .synthetic.phantom import make_bilateral_case

    out = Path(a.out)
    if a.kind == "healthy":
        if a.cases:
            cases = [Path(p) for p in a.cases]
            dem = None
        else:
            cases, dem = [], {}
            for i in range(a.n):
                s = a.seed + i
                p = make_bilateral_case(out / "cases" / f"healthy_{s}", seed=s, injury="intact")
                cases.append(p)
                pr = json.loads((p / "truth" / "truth.json").read_text())["params_injured"]
                # synthetic weak covariate correlated with knee size (stands in for e.g. height)
                dem[f"SYN{s:04d}"] = {"height_cm": pr["scale"] * 170 + float(np.random.default_rng(s).normal(0, 5))}
        rep = healthy_bilateral_holdout(cases, out, demographics=dem)
    else:
        cases = [make_bilateral_case(out / "cases" / f"inj_{k}_{a.seed + i}", seed=a.seed + i, injury=k)
                 for i, k in enumerate((a.injuries or "midsubstance,proximal,distal,partial,chronic_absent").split(","))]
        rep = synthetic_injury_experiment(cases, out)
    md = markdown_table(rep)
    (out / f"{rep['experiment']}.md").write_text(md)
    print(md)


def cmd_evaluate(a):
    import numpy as np

    from .evaluation.metrics import segmentation_report
    from .imaging.volume import load_any

    p, g = load_any(a.pred), load_any(a.gt)
    ig = g.data == 255
    out = {}
    for code in a.labels:
        out[code] = segmentation_report(p.data == code, g.data == code, g.spacing, ignore=ig)
    _p(out)


def cmd_serve(a):
    import uvicorn

    from .ui.server import create_app

    print(f"bilacl research interface on http://{a.host}:{a.port}  (local only; imaging is not sent anywhere)")
    uvicorn.run(create_app(Path(a.workspace)), host=a.host, port=a.port, log_level="warning")


# ---------------------------------------------------------------- parser


def build_parser():
    P = argparse.ArgumentParser(prog="bilacl", description="Bilateral knee MRI ACL anatomy estimation and investigational scaffold research platform")
    sp = P.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("hardware", help="report compute resources")
    s.set_defaults(f=cmd_hardware)
    s = sp.add_parser("registry", help="show the dataset registry")
    s.add_argument("--markdown", action="store_true")
    s.set_defaults(f=cmd_registry)
    s = sp.add_parser("acquire", help="dry-run acquisition report or permitted public download")
    s.add_argument("action", choices=["report", "download"])
    s.add_argument("dataset", nargs="?")
    s.add_argument("--data-root", default="data")
    s.add_argument("--confirm", action="store_true")
    s.add_argument("--allow-large", action="store_true")
    s.add_argument("--sha256")
    s.add_argument("--no-remote", action="store_true")
    s.add_argument("--out")
    s.set_defaults(f=cmd_acquire)
    s = sp.add_parser("inspect", help="inspect a local dataset installation with its adapter")
    s.add_argument("adapter")
    s.add_argument("root")
    s.set_defaults(f=cmd_inspect)
    s = sp.add_parser("manifest", help="build / split / audit manifests")
    s.add_argument("action", choices=["build", "split", "audit", "summary", "subset"])
    s.add_argument("--sources", nargs="*", default=[], help="adapter=root pairs")
    s.add_argument("--manifest")
    s.add_argument("--out")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-subjects", type=int, default=5)
    s.set_defaults(f=cmd_manifest)
    s = sp.add_parser("dicom", help="discover/validate DICOM series, optionally convert to NIfTI")
    s.add_argument("dir")
    s.add_argument("--out", required=True)
    s.add_argument("--prefer")
    s.add_argument("--convert", action="store_true")
    s.set_defaults(f=cmd_dicom)
    s = sp.add_parser("preprocess", help="bias correction, resampling, normalisation, QC images")
    s.add_argument("image")
    s.add_argument("--labels")
    s.add_argument("--out", required=True)
    s.add_argument("--spacing", type=float, nargs=3)
    s.add_argument("--order", type=int, default=1)
    s.add_argument("--n4", action="store_true")
    s.set_defaults(f=cmd_preprocess)
    s = sp.add_parser("annotate", help="3D Slicer import/export, reader agreement, review")
    s.add_argument("action", choices=["color-table", "export-slicer", "import-slicer", "agreement", "review"])
    s.add_argument("--labels")
    s.add_argument("--seg")
    s.add_argument("--reference")
    s.add_argument("--out")
    s.add_argument("--masks", nargs="*")
    s.add_argument("--exam-id", default="unknown")
    s.add_argument("--annotator", default="unknown")
    s.add_argument("--role", default="unknown")
    s.add_argument("--version", type=int, default=1)
    s.add_argument("--parent")
    s.add_argument("--annotated", type=int, nargs="*")
    s.add_argument("--source", default="manual")
    s.add_argument("--model-id")
    s.add_argument("--tear-location", default="unknown")
    s.add_argument("--record")
    s.add_argument("--status")
    s.add_argument("--reviewer")
    s.set_defaults(f=cmd_annotate)
    s = sp.add_parser("synth", help="generate SYNTHETIC bilateral phantom cases")
    s.add_argument("--out", default="workspace/cases")
    s.add_argument("--n", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--injury", default="midsubstance", choices=["intact", "proximal", "midsubstance", "distal", "partial", "chronic_absent"])
    s.add_argument("--side", default="right", choices=["left", "right"])
    s.set_defaults(f=cmd_synth)
    s = sp.add_parser("tasks", help="report which training tasks are runnable or blocked for a manifest")
    s.add_argument("--manifest", required=True)
    s.set_defaults(f=cmd_tasks)
    s = sp.add_parser("train", help="train a segmentation baseline from a YAML config")
    s.add_argument("--config", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--out")
    s.set_defaults(f=cmd_train)
    s = sp.add_parser("infer", help="segmentation inference (outputs unreviewed predictions)")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_infer)
    s = sp.add_parser("register", help="bilateral registration only")
    s.add_argument("case")
    s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_register)
    s = sp.add_parser("run", help="full case pipeline: registration, estimation, uncertainty, abstention, scaffold, report")
    s.add_argument("case")
    s.add_argument("--out", required=True)
    s.add_argument("--scaffold-config")
    s.add_argument("--mechanics-config")
    s.add_argument("--ensemble", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(f=cmd_run)
    s = sp.add_parser("review", help="record expert review of a run's anatomy estimate")
    s.add_argument("run")
    s.add_argument("--status", required=True, choices=["unreviewed", "in_review", "corrected", "approved", "rejected"])
    s.add_argument("--reviewer", required=True)
    s.add_argument("--select", choices=["contralateral_mirror", "remnant_reconciled"])
    s.add_argument("--note", default="")
    s.set_defaults(f=cmd_review)
    s = sp.add_parser("experiment", help="anatomy-estimation experiments (healthy hold-out / synthetic injury)")
    s.add_argument("kind", choices=["healthy", "injury"])
    s.add_argument("--out", required=True)
    s.add_argument("--cases", nargs="*")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--seed", type=int, default=100)
    s.add_argument("--injuries")
    s.set_defaults(f=cmd_experiment)
    s = sp.add_parser("evaluate", help="segmentation metrics (Dice, ASSD, HD95, volume) with UNKNOWN ignored")
    s.add_argument("--pred", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--labels", type=int, nargs="+", required=True)
    s.set_defaults(f=cmd_evaluate)
    s = sp.add_parser("serve", help="launch the local research interface")
    s.add_argument("--workspace", default="workspace")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(f=cmd_serve)
    return P


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
