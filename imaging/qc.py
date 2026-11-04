"""Annotation quality control: review flags, not proof that a segmentation is anatomically or medically correct.

Every check returns status pass | warn | fail | not_evaluated and, where possible, a location (voxel + world) the
viewer can navigate to. No score is computed: model confidence (when the model reports one) is shown as model
confidence, separately from any validation result.
"""
import numpy as np
from scipy import ndimage as ndi

from .volumes import chk

CATEGORIES = [("file", "File and metadata"), ("alignment", "Image–mask alignment"),
              ("geometry", "Segmentation geometry"), ("review", "Human review")]
JUMP = 0.6            # relative slice-to-slice area change flagged as abrupt
MIN_COMPONENT = 0.05  # components smaller than 5% of the largest are reported as extra fragments


def _loc(A, ijk):
    ijk = [float(x) for x in ijk]
    w = (np.asarray(A) @ np.r_[ijk, 1.0])[:3]
    return {"voxel": [round(x, 1) for x in ijk], "world": [round(float(x), 2) for x in w]}


def run(volume, mask=None, mask_arr=None):
    A = volume["affine"]
    out = []
    # ------------------------------------------------ file and metadata (from import validation)
    for c in volume.get("checks", []):
        if c.get("category", "file") == "file":
            out.append(dict(c, category="file"))
    if not volume["spacing_known"]:
        out.append(chk("spatial_metadata", "file", "fail", "Missing physical spacing: millimetre measurements are unavailable "
                       "and mask volumes cannot be compared with anatomy.", "Use DICOM or NIfTI input."))
    if mask is None:
        for cat in ("alignment", "geometry", "review"):
            out.append(chk("no_mask", cat, "not_evaluated", "No mask selected for this volume."))
        return _wrap(out)
    # ------------------------------------------------ alignment
    for c in mask.get("checks", []):
        if c.get("category") == "alignment":
            out.append(dict(c))
    if list(mask_arr.shape) != list(volume["shape"]):
        out.append(chk("dims", "alignment", "fail", "Mask dimensions %s differ from the volume %s."
                       % (list(mask_arr.shape), volume["shape"])))
        return _wrap(out)
    out.append(chk("dims", "alignment", "pass", "Mask dimensions match the volume %s." % (volume["shape"],)))
    labs = {int(l["value"]): l["name"] for l in mask["labels"]}
    present = sorted(int(v) for v in np.unique(mask_arr) if v)
    undefined = [v for v in present if v not in labs]
    out.append(chk("label_table", "alignment", "warn" if undefined else "pass",
                   "Values %s are not in the label table." % undefined if undefined else
                   "All mask values (%s) are named labels." % (", ".join("%d %s" % (v, labs[v]) for v in present) or "none")))
    mm = (mask.get("model") or {}).get("mapping")
    if mm:
        st = "pass" if mm.get("round_trip_max_vox", 1) <= 0.51 and not mm.get("outside_points") else "warn"
        out.append(chk("model_transform", "alignment", st,
                       "Model output mapped from %s into the volume: round-trip error %.2f voxel, %d outline point(s) outside "
                       "the image." % (mm.get("source_space", "model space"), mm.get("round_trip_max_vox", float("nan")),
                                       mm.get("outside_points", 0)), mm.get("note")))
    # ------------------------------------------------ geometry
    if not present:
        out.append(chk("empty", "geometry", "fail", "The mask is empty (no labelled voxels)."))
    for v in sorted(labs):
        name = labs[v]
        m = mask_arr == v
        n = int(m.sum())
        if n == 0:
            out.append(chk("empty_%d" % v, "geometry", "warn", "Label '%s' has no voxels." % name, label=name))
            continue
        idx = np.argwhere(m)
        border = np.any(idx == 0) or np.any(idx >= np.array(mask_arr.shape) - 1)
        if border:
            b = idx[np.any((idx == 0) | (idx >= np.array(mask_arr.shape) - 1), axis=1)][0]
            out.append(dict(chk("border_%d" % v, "geometry", "warn", "Label '%s' touches the image border: the structure may "
                                "extend outside the field of view." % name, label=name), location=_loc(A, b)))
        ax = int(volume.get("native_axis", 2))
        occupied = np.unique(idx[:, ax])
        if len(occupied) == 1:
            out.append(dict(chk("single_slice_%d" % v, "geometry", "not_evaluated",
                                "Label '%s' is on a single slice (%d): it is a 2D outline, so 3D connectivity and slice-to-slice "
                                "checks do not apply." % (name, occupied[0]), label=name),
                            location=_loc(A, idx.mean(0))))
            continue
        lab, ncomp = ndi.label(m, structure=np.ones((3, 3, 3)))
        sizes = ndi.sum(m, lab, range(1, ncomp + 1))
        big = sizes.max()
        if ncomp > 1:
            small = int(np.argsort(sizes)[0]) + 1             # report the smallest part (most likely a stray fragment)
            com = ndi.center_of_mass(m, lab, small)
            out.append(dict(chk("components_%d" % v, "geometry", "warn",
                                "Label '%s' has %d disconnected parts (largest %d voxels, smallest %d)." %
                                (name, ncomp, int(big), int(sizes[small - 1])), label=name), location=_loc(A, com)))
        else:
            out.append(chk("components_%d" % v, "geometry", "pass", "Label '%s' is one connected region." % name, label=name))
        area = np.array([m.take(k, axis=ax).sum() for k in range(mask_arr.shape[ax])], float)
        ks = np.nonzero(area)[0]
        gaps = [k for k in range(ks.min(), ks.max() + 1) if area[k] == 0]
        jumps = []
        for k in range(ks.min() + 1, ks.max() + 1):
            a0, a1 = area[k - 1], area[k]
            if a0 and a1 and abs(a1 - a0) / max(a0, a1) > JUMP:
                jumps.append(k)
        if gaps:
            k = gaps[0]
            p = idx.mean(0)
            p[ax] = k
            out.append(dict(chk("slice_gap_%d" % v, "geometry", "warn", "Label '%s' is missing on %d slice(s) inside its extent "
                                "(first: slice %d along axis %s)." % (name, len(gaps), k, "ijk"[ax]), label=name), location=_loc(A, p)))
        if jumps:
            k = jumps[0]
            sl = np.argwhere(m.take(k, axis=ax))
            p = np.insert(sl.mean(0), ax, k)
            out.append(dict(chk("slice_jump_%d" % v, "geometry", "warn", "Label '%s' changes area abruptly between slices at %d "
                                "slice(s) (> %d%% change; first at slice %d)." % (name, len(jumps), int(JUMP * 100), k), label=name),
                            location=_loc(A, p)))
        if not gaps and not jumps:
            out.append(chk("slices_%d" % v, "geometry", "pass", "Label '%s': no gaps or abrupt area changes across %d slices."
                           % (name, len(ks)), label=name))
    # ------------------------------------------------ human review
    rv = mask.get("review") or {}
    origin = mask.get("origin")
    if rv.get("status") == "reviewed":
        out.append(chk("review", "review", "pass", "Marked reviewed (%s)%s." % (rv.get("at", ""),
                       "; manually edited from the model prediction" if mask.get("edited") else "")))
    elif rv.get("status") == "rejected":
        out.append(chk("review", "review", "fail", "Rejected by the reviewer."))
    else:
        out.append(chk("review", "review", "warn", "Not yet reviewed (%s)." % {"model": "model prediction", "edited": "edited copy",
                                                                                 "import": "imported mask", "manual": "manual drawing"}.get(origin, origin)))
    conf = (mask.get("model") or {}).get("confidence")
    if conf is not None:
        out.append(chk("model_confidence", "review", "not_evaluated", "Model-reported confidence %s is the model's own estimate "
                       "(e.g. predicted IoU); it is not a measured accuracy for this scan." % conf))
    return _wrap(out)


def _wrap(checks):
    cats = []
    for cid, title in CATEGORIES:
        items = [c for c in checks if c.get("category") == cid]
        worst = "not_evaluated"
        for s in ("fail", "warn", "pass"):
            if any(c["status"] == s for c in items):
                worst = s
                break
        cats.append({"id": cid, "title": title, "status": worst, "checks": items})
    return {"categories": cats, "note": "Review flags only: passing checks do not show that a segmentation is correct."}
