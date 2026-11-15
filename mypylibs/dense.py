"""Step 6: dense structure from motion.

With every camera posed (step 5), each registered frame f is paired with the next
frame g that has rotated at least `min_angle` deg. The pair is stereo-rectified
with the known relative pose, SGBM gives a disparity per pixel, and every valid
pixel is back-projected into the world frame.

Averaging across frames: world points are binned into voxels; each voxel's
position and colour is the mean of what landed in it, and a voxel is kept only if
at least `min_views` different pairs put a point there (one-off SGBM noise dies;
min_views=3 vs 2: a 36k -> 15k points, b 250k -> 60k, visibly cleaner in both).
"""
import cv2
import numpy as np


def _angle(R):
    return np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))


def relative(cams, f, g):
    """Pose of camera g relative to camera f: x_g = R x_f + t."""
    (Rf, tf), (Rg, tg) = cams[f], cams[g]
    R = Rg @ Rf.T
    return R, tg - R @ tf


def dense_pair(img1, img2, R, t, K, zmin, zmax, stride=2):
    """Points (N,3) in camera-1 coords and RGB colours (N,3) from one posed pair.

    Only depths in [zmin, zmax] are searched (sets the SGBM disparity range) and kept.
    Returns None when the motion is mostly vertical (SGBM only searches along x).
    """
    h, w = img1.shape[:2]
    t = np.asarray(t, np.float64).reshape(3, 1)
    # alpha=-1 keeps the original focal length; alpha=0 crops to valid pixels and with
    # any forward motion zooms in up to 10x (f 7200 vs 623), pushing disparities off-image.
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, None, K, None, (w, h), R, t, alpha=-1)
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        return None
    if P2[0, 3] > 0:  # camera 2 is left of camera 1: swap so disparities are positive
        out = dense_pair(img2, img1, R.T, -R.T @ t, K, zmin, zmax, stride)
        if out is None:
            return None
        X2, c = out
        return (X2 - t.ravel()) @ R, c  # back to camera-1 coords: R^T (X2 - t)

    r1 = cv2.remap(img1, *cv2.initUndistortRectifyMap(K, None, R1, P1, (w, h), cv2.CV_32FC1), cv2.INTER_LINEAR)
    r2 = cv2.remap(img2, *cv2.initUndistortRectifyMap(K, None, R2, P2, (w, h), cv2.CV_32FC1), cv2.INTER_LINEAR)
    fB = -P2[0, 3]  # focal * baseline, disparity = fB / z
    dmin = max(0, int(fB / zmax))
    num = min(max(16, int(np.ceil((fB / zmin - dmin + 1) / 16)) * 16), (w - dmin) // 16 * 16)
    if num < 16:
        return None
    block = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=dmin, numDisparities=num, blockSize=block,
                                 P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(r1, r2).astype(np.float32) / 16
    X = cv2.reprojectImageTo3D(disp, Q)
    valid = (disp >= dmin) & (X[..., 2] > zmin) & (X[..., 2] < zmax)
    sub = np.zeros_like(valid)
    sub[::stride, ::stride] = True  # ponytail: plain pixel subsampling, neighbours add little after voxel averaging
    valid &= sub
    return X[valid] @ R1, r1[valid][:, ::-1].copy()  # rectified -> camera-1 coords: R1^T X


def voxel_average(X, colours, pair_id, voxel, min_views=2):
    """Mean point and colour per voxel, keeping voxels hit by >= min_views pairs."""
    key = np.floor(X / voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n)
    views = np.bincount(np.unique(np.c_[inv, pair_id], axis=0)[:, 0], minlength=n)
    mean = lambda v: np.stack([np.bincount(inv, v[:, d], n) / cnt for d in range(3)], 1)
    keep = views >= min_views
    return mean(X)[keep], mean(colours.astype(np.float64))[keep].round().astype(np.uint8)


def densify(frames, cams, pts, res, K, min_angle=5.0, min_views=3, voxel_div=300):
    """Step 6 over a whole video. Returns (X (N,3) world points, RGB colours (N,3)).

    frames: undistorted BGR frames; cams/pts/res: step 5 output and step 2 tracks.
    Depth range per frame comes from the sparse points it sees (2nd-98th percentile,
    widened 20%); voxel size = sparse cloud extent / voxel_div.
    """
    fs = sorted(cams)
    sparse = np.array(list(pts.values()))
    voxel = (np.percentile(sparse, 95, axis=0) - np.percentile(sparse, 5, axis=0)).max() / voxel_div
    seen: dict[int, list[int]] = {}
    for tid in pts:
        for f in res.tracks[tid]:
            seen.setdefault(f, []).append(tid)

    Xs, Cs, ids = [], [], []
    for f in fs:
        g = next((g for g in fs if g > f and _angle(relative(cams, f, g)[0]) >= min_angle), None)
        if g is None:
            break
        if len(seen.get(f, [])) < 20:
            continue
        Rf, tf = cams[f]
        z = (np.array([pts[t] for t in seen[f]]) @ Rf.T + tf)[:, 2]
        zmin, zmax = np.percentile(z, [2, 98]) * [0.8, 1.2]
        if zmin <= 0:
            continue
        out = dense_pair(frames[f], frames[g], *relative(cams, f, g), K, zmin, zmax)
        if out is None or not len(out[0]):
            continue
        Xs.append((out[0] - tf) @ Rf)  # camera f -> world: R^T (X - t)
        Cs.append(out[1])
        ids.append(np.full(len(out[0]), f))
    if not Xs:
        return np.empty((0, 3)), np.empty((0, 3), np.uint8)
    X, C, ids = np.concatenate(Xs), np.concatenate(Cs), np.concatenate(ids)
    # Clip to the sparse cloud's box (2-98th pct, +10%). In b the sparse points are only
    # the moving object, while SGBM also "triangulates" the static background into junk.
    lo, hi = np.percentile(sparse, 2, axis=0), np.percentile(sparse, 98, axis=0)
    pad = 0.1 * (hi - lo)
    inside = ((X > lo - pad) & (X < hi + pad)).all(1)
    return voxel_average(X[inside], C[inside], ids[inside], voxel, min_views)


if __name__ == '__main__':
    # Self-check: random-texture plane z=5 seen by two cameras, the dense points must
    # land on the plane, whichever way the camera moved (exercises the swap branch).
    rng = np.random.default_rng(0)
    K = np.array([[500., 0, 180], [0, 500, 320], [0, 0, 1]])
    tex = cv2.GaussianBlur(rng.integers(0, 255, (640, 360, 3), dtype=np.uint8), (5, 5), 0)
    R, _ = cv2.Rodrigues(np.array([0, 0.02, 0.]))
    n, d = np.array([[0, 0, 1.]]), 5.0
    for tx in (-0.4, 0.4):
        t = np.array([[tx], [0], [0.]])
        H = K @ (R + t @ n / d) @ np.linalg.inv(K)  # plane-induced homography cam1 -> cam2
        img2 = cv2.warpPerspective(tex, H, (360, 640))
        X, c = dense_pair(tex, img2, R, t, K, 3.0, 8.0)
        assert len(X) > 5000, (tx, len(X))
        assert np.median(np.abs(X[:, 2] - d)) < 0.05, (tx, np.median(X[:, 2]))
    # voxel averaging: two pairs agreeing -> kept; a lone point -> dropped
    Xv, cv = voxel_average(np.array([[0.1, 0, 0], [0.12, 0, 0], [5, 5, 5]]),
                           np.array([[10, 0, 0], [20, 0, 0], [0, 0, 0]]), np.array([0, 1, 0]), 1.0)
    assert len(Xv) == 1 and np.allclose(Xv[0], [0.11, 0, 0]) and cv[0, 0] == 15
    print('ok')
