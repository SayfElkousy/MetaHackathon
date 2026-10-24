"""Local research interface (FastAPI). Binds to 127.0.0.1 by default; imaging never leaves the machine.

Every payload carries evidence/model/review status so the front end can badge
synthetic demonstrations, untrained/smoke-test models, unreviewed estimates and
expert-reviewed results distinctly, and show the specific missing input when
data or weights are absent.
"""
from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

STATIC = Path(__file__).parent / "static"


def create_app(workspace: Path) -> FastAPI:
    ws = Path(workspace).resolve()
    runs_root = ws / "runs"
    app = FastAPI(title="bilacl research interface")

    # ------------------------------------------------------------ helpers
    def cases() -> dict:
        out = {}
        for p in sorted(ws.rglob("case.json")):
            if "runs" in p.parts:
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            out[d["case_id"]] = p.parent
        return out

    def case_dir(cid) -> Path:
        c = cases()
        if cid not in c:
            raise HTTPException(404, f"unknown case {cid}")
        return c[cid]

    def run_dir(cid) -> Path:
        return runs_root / cid

    @lru_cache(maxsize=48)
    def _load(path: str, mtime: float):
        from ..imaging.volume import load_nifti

        return load_nifti(path)

    def load(path: Path):
        if not path.exists():
            raise HTTPException(404, f"missing input: {path.name}")
        return _load(str(path), path.stat().st_mtime)

    def volume_path(cid, name) -> Path:
        cd, rd = case_dir(cid), run_dir(cid)
        c = json.loads((cd / "case.json").read_text())
        m = {
            "injured_image": cd / c["injured"]["image"], "injured_labels": cd / c["injured"]["labels"],
            "contralateral_image": cd / c["contralateral"]["image"], "contralateral_labels": cd / c["contralateral"]["labels"],
            "mirrored_image": rd / "inspection" / "contralateral_mirrored_registered.nii.gz",
            "mirrored_labels": rd / "inspection" / "contralateral_labels_mirrored_registered.nii.gz",
        }
        for f in (rd / "estimates").glob("*.nii.gz") if (rd / "estimates").exists() else []:
            m["est_" + f.name.replace(".nii.gz", "")] = f
        corr = cd / "injured" / "labels_corrected.nii.gz"
        if corr.exists():
            m["injured_labels_corrected"] = corr
        if name not in m:
            raise HTTPException(404, f"unknown volume {name}")
        return m[name]

    # ------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    # ------------------------------------------------------------ inventory
    @app.get("/api/inventory")
    def inventory():
        from ..acquisition import acquisition_report, load_registry

        reg = load_registry()
        rep = acquisition_report(str(ws.parent / "data"), probe_remote=False, registry=reg)
        rows = []
        for k, d in reg.items():
            loc = rep["datasets"][k].get("local", {})
            states = [v.get("state") for v in d.values() if isinstance(v, dict) and "state" in v]
            rows.append({"key": k, "name": d.get("name"), "source": d.get("official_source"), "verified_on": d.get("verified_on"),
                         "page_readable": d.get("page_readable"), "download_mode": d.get("download_mode"),
                         "fields_verified": states.count("verified"), "fields_total": len(states),
                         "installed": bool(loc.get("exists")), "blocking": loc.get("blocking", []),
                         "role": d.get("role", []), "labels": (d.get("labels") or {}).get("value"),
                         "missing_labels": d.get("missing_labels_blocking", []), "limitations": d.get("population_limitations"),
                         "requirement": rep["datasets"][k].get("requirement"), "status": d.get("status")})
        return {"datasets": rows, "disk_free_gb": rep["disk_free_gb"]}

    @app.get("/api/models")
    def models():
        out = []
        for p in sorted(ws.rglob("run.json")):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if "model_status" in d:
                out.append({"path": str(p.parent.relative_to(ws)), "task": d.get("config", {}).get("task") or d.get("task"),
                            "model_status": d["model_status"], "best_val": d.get("best_val_mean_dice", d.get("best_val_auroc")),
                            "train_exams": len(d.get("train_exams", [])), "note": d.get("note", ""), "device": d.get("device")})
        return {"models": out, "validated_weights_available": any(m["model_status"] == "trained" for m in out)}

    @app.get("/api/experiments")
    def experiments():
        out = []
        for p in sorted(ws.rglob("*.json")):
            if p.name in ("healthy_bilateral_holdout.json", "synthetic_injury.json"):
                d = json.loads(p.read_text())
                out.append({"path": str(p.relative_to(ws)), "experiment": d["experiment"], "banner": d.get("banner"),
                            "evidence_level": d.get("evidence_level"), "summary": d["summary"]})
        return {"experiments": out}

    # ------------------------------------------------------------ cases
    @app.get("/api/cases")
    def list_cases():
        out = []
        for cid, p in cases().items():
            d = json.loads((p / "case.json").read_text())
            rp = run_dir(cid) / "pipeline.json"
            out.append({"case_id": cid, "synthetic": d.get("synthetic", False), "injured_side": d["injured"]["side"],
                        "contralateral_side": d["contralateral"]["side"], "tear_location": d["injured"].get("tear_location"),
                        "has_run": rp.exists(), "path": str(p.relative_to(ws))})
        return {"cases": out}

    @app.get("/api/case/{cid}")
    def get_case(cid: str):
        cd = case_dir(cid)
        d = json.loads((cd / "case.json").read_text())
        rd = run_dir(cid)
        res = json.loads((rd / "pipeline.json").read_text()) if (rd / "pipeline.json").exists() else None
        from ..pipeline import load_review

        vols = {}
        for n in ("injured_image", "contralateral_image"):
            v = load(volume_path(cid, n))
            vols[n] = v.summary()
        avail = ["injured_image", "contralateral_image", "injured_labels", "contralateral_labels"]
        if (rd / "inspection").exists():
            avail += ["mirrored_image", "mirrored_labels"]
        if (rd / "estimates").exists():
            avail += ["est_" + f.name.replace(".nii.gz", "") for f in sorted((rd / "estimates").glob("*.nii.gz"))]
        if (cd / "injured" / "labels_corrected.nii.gz").exists():
            avail.append("injured_labels_corrected")
        missing = []
        if not res:
            missing.append("pipeline not yet run for this case (press Run pipeline)")
        lm = cd / d["injured"].get("landmarks", "x")
        if not lm.exists():
            missing.append("injured-side landmarks (registration accuracy cannot be independently checked)")
        if d["contralateral"].get("contralateral_confirmed_normal") is not True and not d.get("synthetic"):
            missing.append("expert confirmation that the contralateral knee is normal")
        return {"case": d, "run": res, "review": load_review(rd), "volumes": vols, "available": avail, "missing": missing}

    @app.get("/api/slice")
    def slice_png(cid: str, vol: str, axis: int = 0, idx: int = -1, overlay: str = "", alpha: float = 0.45, cross: str = ""):
        from PIL import Image

        from ..annotation.schema import LABEL_COLORS, Label

        v = load(volume_path(cid, vol))
        d = v.data.astype(np.float32)
        if idx < 0:
            idx = d.shape[axis] // 2
        idx = int(np.clip(idx, 0, d.shape[axis] - 1))
        sl = np.take(d, idx, axis=axis).T[::-1]  # second remaining axis vertical, increasing upward
        lo, hi = np.percentile(sl, [1, 99.5]) if sl.size else (0, 1)
        g = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.stack([g] * 3, -1)
        if overlay:
            for ov in overlay.split(","):
                o = load(volume_path(cid, ov))
                if o.data.shape[:3] != v.data.shape[:3]:
                    continue  # never overlay across different grids
                osl = np.take(o.data, idx, axis=axis).T[::-1]
                if ov.startswith("est_probability"):
                    p = osl.astype(np.float32)
                    col = np.array([1.0, 0.2, 0.8])
                    a = (alpha * p)[..., None]
                    rgb = rgb * (1 - a) + col * a
                elif ov.startswith("est_"):
                    m = osl > 0
                    edge = m & ~_erode2d(m)
                    rgb[edge] = [0.2, 1.0, 0.4] if "remnant" in ov else [0.3, 0.8, 1.0]
                else:
                    for code in np.unique(osl):
                        if code == 0:
                            continue
                        try:
                            col = np.array(LABEL_COLORS.get(Label(int(code)), (200, 200, 200))) / 255
                        except ValueError:
                            col = np.array([1, 1, 1])
                        m = osl == code
                        rgb[m] = rgb[m] * (1 - alpha) + col * alpha
        if cross:
            a, b = (int(x) for x in cross.split(","))
            H = rgb.shape[0]
            if 0 <= a < rgb.shape[1]:
                rgb[:, a] = rgb[:, a] * 0.4 + np.array([1, 0.85, 0]) * 0.6
            if 0 <= b < rgb.shape[0]:
                rgb[H - 1 - b, :] = rgb[H - 1 - b, :] * 0.4 + np.array([1, 0.85, 0]) * 0.6
        sp = v.spacing
        other = [i for i in range(3) if i != axis]
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        w = int(round(img.width * sp[other[0]] / min(sp) * 2))
        h = int(round(img.height * sp[other[1]] / min(sp) * 2))
        img = img.resize((w, h), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"X-Index": str(idx), "X-Shape": ",".join(map(str, d.shape[:3])),
                                                                           "Cache-Control": "no-store"})

    @app.get("/api/case/{cid}/meshes")
    def meshes(cid: str):
        import trimesh

        from ..annotation.schema import Label
        from ..registration.bilateral import surface_mesh

        rd = run_dir(cid)
        cache = rd / "inspection" / "meshes.json"
        src = volume_path(cid, "injured_labels")
        if cache.exists() and cache.stat().st_mtime > max(src.stat().st_mtime, (rd / "pipeline.json").stat().st_mtime if (rd / "pipeline.json").exists() else 0):
            return JSONResponse(json.loads(cache.read_text()))
        lab = load(src)
        out = {"frame": "RAS+ mm, injured knee", "objects": []}

        def add(name, mask, aff, color, level, opacity=1.0, step=2):
            if mask.sum() < 20:
                return
            m = surface_mesh(mask, aff, step=step)
            m = _decimate(m, 30000)
            out["objects"].append({"name": name, "evidence": level, "color": color, "opacity": opacity,
                                   "vertices": np.round(m.vertices, 2).ravel().tolist(), "faces": m.faces.ravel().tolist()})

        add("femur", lab.data == Label.FEMUR, lab.affine, "#d9cfa0", "observed (annotation)")
        add("tibia", lab.data == Label.TIBIA, lab.affine, "#b9b0dc", "observed (annotation)")
        add("ACL remnants", np.isin(lab.data, [Label.ACL_REMNANT_FEMORAL, Label.ACL_REMNANT_TIBIAL, Label.ACL_FRAGMENT_OTHER]), lab.affine,
            "#ff9f1a", "observed (annotation)", step=1)
        add("ACL (intact fibres)", lab.data == Label.ACL_INTACT, lab.affine, "#e8503a", "observed (annotation)", step=1)
        for est, col in (("remnant_reconciled", "#3ddc84"), ("contralateral_mirror", "#4fb3ff")):
            p = rd / "estimates" / f"{est}.nii.gz"
            if p.exists():
                e = load(p)
                add(f"estimate: {est}", e.data > 0, e.affine, col, "estimated", 0.35, step=1)
        sc = next(iter((rd / "scaffold").glob("*scaffold_candidate.stl")), None) if (rd / "scaffold").exists() else None
        if sc:
            m = trimesh.load(sc)
            m = _decimate(m, 60000)
            out["objects"].append({"name": "scaffold candidate", "evidence": "hypothesized (investigational geometry)", "color": "#6aa0ff",
                                   "opacity": 0.55, "vertices": np.round(m.vertices, 2).ravel().tolist(), "faces": m.faces.ravel().tolist()})
            pw = next(iter((rd / "scaffold").glob("*hypothesized_guidance_pathways.json")), None)
            if pw:
                out["pathways"] = json.loads(pw.read_text())["polylines"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
        return out

    @app.post("/api/case/{cid}/run")
    def run(cid: str, body: dict = Body(default={})):
        from ..pipeline import load_config, run_case

        root = Path(__file__).resolve().parents[3]
        sc = load_config(root / "configs" / "scaffold_demo.yaml") if body.get("scaffold", True) else None
        mc = load_config(root / "configs" / "mechanics_demo.yaml") if body.get("mechanics", True) else None
        logs = []
        res = run_case(case_dir(cid), run_dir(cid), sc, mc, ensemble_size=int(body.get("ensemble", 6)), log=logs.append)
        return {"status": res["status"], "log": logs}

    @app.post("/api/case/{cid}/review")
    def review(cid: str, body: dict = Body(...)):
        from ..pipeline import set_review

        if not body.get("reviewer"):
            raise HTTPException(400, "reviewer identity required")
        return set_review(run_dir(cid), body["status"], body["reviewer"], body.get("select"), body.get("note", ""))

    @app.post("/api/case/{cid}/correction")
    async def correction(cid: str, file: UploadFile = File(...), annotator: str = Form(...), note: str = Form("")):
        """Import an expert correction (Slicer .seg.nrrd or NIfTI) for the injured knee as a new annotation version."""
        import tempfile

        from ..annotation.schema import AnnotationRecord, read_slicer_seg_nrrd
        from ..imaging.preprocess import resample_like
        from ..imaging.volume import PhysicalStatus, Volume, load_any, save_nifti
        from ..provenance import ReviewStatus

        cd = case_dir(cid)
        ref = load(volume_path(cid, "injured_labels"))
        suffix = ".seg.nrrd" if file.filename.endswith(".nrrd") else ".nii.gz"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as t:
            t.write(await file.read())
            tp = t.name
        if suffix == ".seg.nrrd":
            lab, aff, _ = read_slicer_seg_nrrd(tp)
            v = Volume(lab, aff, PhysicalStatus(True, "slicer"))
        else:
            v = load_any(tp)
        v = resample_like(v, ref, kind="label")
        save_nifti(v, cd / "injured" / "labels_corrected.nii.gz", np.uint8)
        base = cd / "injured" / "labels.annotation.json"
        prev = AnnotationRecord.load(base) if base.exists() else None
        rec = AnnotationRecord(annotation_id=f"{cid}_corrected", exam_id=prev.exam_id if prev else cid, image_ref="injured/image.nii.gz",
                               annotator=annotator, annotator_role="expert", version=(prev.version + 1) if prev else 1,
                               parent_annotation_id=prev.annotation_id if prev else None,
                               annotated_labels=sorted({int(x) for x in np.unique(v.data)} - {0, 255}), source="manual",
                               review_status=ReviewStatus.CORRECTED, notes=note)
        rec.save(cd / "injured" / "labels_corrected.annotation.json")
        return {"status": "imported", "record": rec.to_dict(),
                "note": "stored as a new version; the pipeline still uses the original labels until the case file is pointed at the correction"}

    @app.get("/api/case/{cid}/report", response_class=HTMLResponse)
    def report(cid: str):
        p = run_dir(cid) / "report.html"
        if not p.exists():
            raise HTTPException(404, "no report yet: run the pipeline first")
        return p.read_text()

    @app.get("/api/case/{cid}/qc/{name}")
    def qc(cid: str, name: str):
        p = run_dir(cid) / "qc" / f"{name}.png"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/case/{cid}/file/{name}")
    def file(cid: str, name: str):
        sd = run_dir(cid) / "scaffold"
        p = sd / name
        if not p.exists() or p.parent != sd:
            raise HTTPException(404)
        return FileResponse(p, filename=name)

    return app


def _decimate(m, n):
    """Display-only decimation; falls back to the full mesh if no simplifier is installed."""
    if len(m.faces) <= n:
        return m
    try:
        return m.simplify_quadric_decimation(face_count=n)
    except Exception:
        return m


def _erode2d(m):
    from scipy import ndimage

    return ndimage.binary_erosion(m)
