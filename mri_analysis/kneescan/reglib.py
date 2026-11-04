import numpy as np, SimpleITK as sitk
ZS=6.0
def load_norm(path):
    v=np.load(path).astype(np.float32)
    p1,p99=np.percentile(v,[1,99.5]); return np.clip((v-p1)/(p99-p1+1e-6),0,1)
def to_img(v):
    img=sitk.GetImageFromArray(np.ascontiguousarray(v)); img.SetSpacing((1.0,1.0,ZS)); return img
def _method(lr, iters, seed, pct):
    R=sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(pct, seed=seed)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(lr, 1e-3, iters, relaxationFactor=0.6)
    R.SetOptimizerScalesFromPhysicalShift(); return R
def _stage1(fixed, moving, seed, fast, scale=None):
    R=_method(1.5, 80 if fast else 120, seed, 0.06)
    init=sitk.CenteredTransformInitializer(fixed, moving, sitk.ScaleVersor3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    if scale is not None:
        init=sitk.ScaleVersor3DTransform(init); init.SetScale((scale, scale, 1.0))
    R.SetInitialTransform(init, inPlace=False)
    R.SetShrinkFactorsPerLevel([4,2]); R.SetSmoothingSigmasPerLevel([2,1]); R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    t1=R.Execute(fixed, moving); return t1, R.GetMetricValue()
def register(fixed, moving, seed=1, fast=False, scales=None):
    """transform maps fixed physical pts -> moving physical pts. scales: try several initial in-plane zooms."""
    if scales:
        best=None
        for sc in scales:
            try: t, m = _stage1(fixed, moving, seed, True, sc)
            except Exception: continue
            if best is None or m < best[1]: best=(t, m, sc)
        t1, m1 = _stage1(fixed, moving, seed, fast, best[2]) if not fast else (best[0], best[1])
    else:
        t1, m1 = _stage1(fixed, moving, seed, fast)
    if fast: return t1, m1
    A=sitk.AffineTransform(3); A.SetCenter((128.0,128.0,ZS*fixed.GetSize()[2]/2))
    R2=_method(1.0, 80, seed, 0.06)
    R2.SetMovingInitialTransform(t1); R2.SetInitialTransform(A, inPlace=True)
    R2.SetShrinkFactorsPerLevel([2,1]); R2.SetSmoothingSigmasPerLevel([1,0.5]); R2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    R2.Execute(fixed, moving); m2=R2.GetMetricValue()
    return sitk.CompositeTransform([t1, A]), m2
def refine_roi(fixed, moving, tr, center_xyz, half=(60,60,4), seed=2):
    """affine refine restricted to a box around center (fixed voxel coords x,y,slice)"""
    size=fixed.GetSize(); m=np.zeros((size[2],size[1],size[0]),np.uint8)
    cx,cy,cz=center_xyz
    z0,z1=int(max(0,cz-half[2])),int(min(size[2],cz+half[2]+1)); y0,y1=int(max(0,cy-half[1])),int(min(size[1],cy+half[1])); x0,x1=int(max(0,cx-half[0])),int(min(size[0],cx+half[0]))
    if z1<=z0 or y1<=y0 or x1<=x0: return tr, None
    m[z0:z1,y0:y1,x0:x1]=1
    mask=sitk.GetImageFromArray(m); mask.CopyInformation(fixed)
    A=sitk.AffineTransform(3); A.SetCenter((float(cx),float(cy),float(cz)*ZS))
    R=_method(0.5, 60, seed, 0.25); R.SetMetricFixedMask(mask)
    R.SetMovingInitialTransform(tr); R.SetInitialTransform(A, inPlace=True)
    R.SetShrinkFactorsPerLevel([1]); R.SetSmoothingSigmasPerLevel([0.7]); R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    try:
        R.Execute(fixed, moving)
    except Exception: return tr, None
    return sitk.CompositeTransform([tr, A]), R.GetMetricValue()
def to_fixed(tr, p):
    q=tr.GetInverse().TransformPoint((float(p[0]),float(p[1]),float(p[2])*ZS))
    return np.array([q[0],q[1],q[2]/ZS])
def inplane_scale(tr, p):
    """local in-plane scale factor (moving px per fixed px) near fixed point p via finite differences"""
    c=np.array([p[0],p[1],p[2]*ZS],float)
    a=np.array(tr.TransformPoint(tuple(c))); bx=np.array(tr.TransformPoint(tuple(c+[10,0,0]))); by=np.array(tr.TransformPoint(tuple(c+[0,10,0])))
    J=np.stack([(bx-a)[:2],(by-a)[:2]],1)/10.0
    return float(np.sqrt(abs(np.linalg.det(J))))
