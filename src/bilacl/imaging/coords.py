"""Coordinate conventions (single source of truth).

Internal convention: every volume carries a 4x4 affine mapping voxel indices
(i, j, k) to RAS+ world coordinates in millimetres (NIfTI / nibabel convention):

    RAS+ : +x -> patient Right, +y -> Anterior, +z -> Superior
    LPS+ : +x -> patient Left,  +y -> Posterior, +z -> Superior   (DICOM patient coordinates)

Conversion flips the signs of x and y and is its own inverse:

    p_RAS = diag(-1, -1, 1) @ p_LPS

Consequence used elsewhere: in RAS+, the lateral side of a RIGHT knee is +x and
the lateral side of a LEFT knee is -x.

tests/test_coords.py checks these statements against nibabel.aff2axcodes and
against SimpleITK (which uses LPS).
"""
from __future__ import annotations

import numpy as np

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
RAS_TO_LPS = LPS_TO_RAS  # the conversion is its own inverse


def lps_affine_to_ras(aff_lps: np.ndarray) -> np.ndarray:
    return LPS_TO_RAS @ np.asarray(aff_lps, float)


def ras_affine_to_lps(aff_ras: np.ndarray) -> np.ndarray:
    return RAS_TO_LPS @ np.asarray(aff_ras, float)


def mirror_matrix(plane_x_mm: float = 0.0) -> np.ndarray:
    """Reflection across the world sagittal plane x = plane_x_mm (RAS mm).

    x' = 2*x0 - x ; y' = y ; z' = z.  det = -1 (handedness flips).
    Used to turn a left knee into a right-knee-like geometry (or vice versa).
    The subsequent rigid registration absorbs the choice of x0, so x0 = 0
    (scanner isocentre plane) is an acceptable documented default.
    """
    M = np.eye(4)
    M[0, 0] = -1.0
    M[0, 3] = 2.0 * plane_x_mm
    return M


def apply_affine(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    return pts @ T[:3, :3].T + T[:3, 3]


def voxel_spacing(affine: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)


def axcodes(affine: np.ndarray) -> tuple[str, str, str]:
    import nibabel as nib

    return tuple(nib.aff2axcodes(affine))


def rigid_from_params(rx: float, ry: float, rz: float, t=(0.0, 0.0, 0.0), center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rotation (radians, XYZ order) about `center` followed by translation t."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    c = np.asarray(center, float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c + np.asarray(t, float)
    return T


def is_rigid(T: np.ndarray, tol: float = 1e-6, allow_reflection: bool = False) -> bool:
    R = T[:3, :3]
    ortho = np.allclose(R.T @ R, np.eye(3), atol=tol)
    d = np.linalg.det(R)
    return ortho and (np.isclose(d, 1.0, atol=tol) or (allow_reflection and np.isclose(d, -1.0, atol=tol)))
