"""Structure from Motion steps.

Step 3: undistort frames, essential matrix E and relative pose (R, t) for every
pair of consecutive frames.
Step 4: triangulate each pair's inlier points into 3D (camera i at the origin)
and measure the reprojection error in both frames.
"""
import cv2
import numpy as np

from mypylibs import features


def load_camera(root='in'):
    """K (3x3) and D (1x5) of the iPhone 15.

    Calibrated in lab2 (camera 02) on 640x360 landscape frames. Our videos are
    360x640 portrait = raw frame rotated 90 deg clockwise (same 1/3 scale), so:
        fx'=fy, fy'=fx, cx'=359-cy, cy'=cx      k1,k2,k3 same, p1'=p2, p2'=-p1
    """
    K = np.loadtxt(f'{root}/K.csv', delimiter=',')
    D = np.loadtxt(f'{root}/D.csv', delimiter=',')
    return K, D


def read_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames = []
    ok, f = cap.read()
    while ok:
        frames.append(f)
        ok, f = cap.read()
    cap.release()
    return frames


def undistort(frames, K, D):
    """Undistorted frames keep K as camera matrix, so later steps use K with D=0."""
    return [cv2.undistort(f, K, D) for f in frames]


def relative_poses(frames, K, cfg=None):
    """For each consecutive pair (i, i+1): dict with E, R, t, inlier points.

    Matching, RANSAC and E come from step 2 (features.run); this adds the pose.
    R, t map camera-i coords to camera-(i+1) coords (x2 = R x1 + t), |t| = 1
    (monocular scale is unknown). 'idx' holds the (M,2) keypoint indices of the
    inliers, so poses line up with feats.tracks. Pairs with too few matches get None.

    Returns (poses, feats), feats being the step-2 FeatureResult.
    """
    feats = features.run(frames, K, cfg)
    poses = []
    for i, (m, E) in enumerate(zip(feats.pair_matches, feats.essentials)):
        if E is None or len(m) < 8:
            poses.append(None)
            continue
        p1 = feats.feats[i].pts[m[:, 0]]
        p2 = feats.feats[i + 1].pts[m[:, 1]]
        n, R, t, mask = cv2.recoverPose(E, p1, p2, K)
        inl = mask.ravel() > 0
        if inl.sum() < 8:  # cheirality check rejected (almost) everything
            poses.append(None)
            continue
        poses.append({'E': E, 'R': R, 't': t, 'p1': p1[inl], 'p2': p2[inl],
                      'idx': m[inl], 'n': int(n)})
    return poses, feats


def triangulate(R, t, p1, p2, K):
    """3D points (N,3) in camera-1 coords, with camera 1 = K[I|0] and camera 2 = K[R|t]."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
    return (Xh[:3] / Xh[3]).T


def reproject(X, R, t, K):
    """Project 3D points (N,3) into a camera with pose x_cam = R X + t. Returns (N,2) pixels."""
    proj, _ = cv2.projectPoints(X, cv2.Rodrigues(R)[0], t.reshape(3, 1).astype(np.float64), K, None)
    return proj.reshape(-1, 2)


def triangulate_pairs(poses, K):
    """Step 4 for every consecutive pair posed in step 3.

    Returns, per pair, a dict with X (3D points in camera-i coords, in units of
    |t| = 1), the matching p1/p2 pixels, and err1/err2: per-point reprojection
    error in pixels in frame i and frame i+1. Points behind either camera are
    dropped. Pairs without a pose stay None.
    """
    out = []
    for pose in poses:
        if pose is None:
            out.append(None)
            continue
        R, t, p1, p2 = pose['R'], pose['t'], pose['p1'], pose['p2']
        X = triangulate(R, t, p1, p2, K)
        z2 = (X @ R.T + t.reshape(1, 3))[:, 2]
        keep = np.isfinite(X).all(axis=1) & (X[:, 2] > 0) & (z2 > 0)
        X, p1, p2 = X[keep], p1[keep], p2[keep]
        err1 = np.linalg.norm(reproject(X, np.eye(3), np.zeros(3), K) - p1, axis=1)
        err2 = np.linalg.norm(reproject(X, R, t, K) - p2, axis=1)
        out.append({'X': X, 'p1': p1, 'p2': p2, 'err1': err1, 'err2': err2})
    return out


# ---------------------------------------------------------------- Step 5: N frames

def _rotate(X, rvec):
    """Rotate each row of X by the matching row of rvec (Rodrigues formula, vectorised)."""
    th = np.linalg.norm(rvec, axis=1, keepdims=True)
    k = np.divide(rvec, th, out=np.zeros_like(rvec), where=th > 0)
    c, s = np.cos(th), np.sin(th)
    return X * c + np.cross(k, X) * s + k * (k * X).sum(1, keepdims=True) * (1 - c)


def _obs(tids, cams, res, frames_used):
    """Observations (frame, point slot, pixel) of tracks `tids` in the frames `frames_used`."""
    of, op, uv = [], [], []
    for j, tid in enumerate(tids):
        for f, k in res.tracks[tid].items():
            if f in frames_used and f in cams:
                of.append(f)
                op.append(j)
                uv.append(res.feats[f].pts[k])
    return of, np.array(op, int), np.array(uv, np.float64).reshape(-1, 2)


def bundle_adjust(cams, pts, res, K, free, max_nfev=30):
    """Refine the poses of `free` frames and every 3D point they see (in place).

    All other registered frames that see those points add their observations but
    stay fixed, which anchors position and scale. Huber loss so the odd wrong
    checkerboard match can't drag the solution. f_scale=2px: at 1px most honest
    residuals fall in the linear part and the solver crawls (4 deg drift over 16
    synthetic frames vs 0.3 deg at 2px).
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    free = [f for f in free if f in cams]
    tids = [tid for tid in pts if any(f in res.tracks[tid] for f in free)]
    if not free or not tids:
        return
    of, op, uv = _obs(tids, cams, res, set(cams))
    order = free + sorted(set(of) - set(free))
    ci = {f: i for i, f in enumerate(order)}
    oc = np.array([ci[f] for f in of])
    par = np.array([np.r_[cv2.Rodrigues(cams[f][0])[0].ravel(), cams[f][1]] for f in order])
    nf = len(free)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def resid(x):
        cp = np.vstack([x[:6 * nf].reshape(-1, 6), par[nf:]])
        Xc = _rotate(x[6 * nf:].reshape(-1, 3)[op], cp[oc, :3]) + cp[oc, 3:]
        z = Xc[:, 2:3]
        z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        return (Xc[:, :2] / z * [fx, fy] + [cx, cy] - uv).ravel()

    x0 = np.r_[par[:nf].ravel(), np.array([pts[t] for t in tids]).ravel()]
    A = lil_matrix((2 * len(oc), x0.size), dtype=int)
    i = np.arange(len(oc))
    m = oc < nf
    for r in (0, 1):
        for d in range(6):
            A[2 * i[m] + r, 6 * oc[m] + d] = 1
        for d in range(3):
            A[2 * i + r, 6 * nf + 3 * op + d] = 1
    x = least_squares(resid, x0, jac_sparsity=A, loss='huber', f_scale=2.0,
                      x_scale='jac', method='trf', max_nfev=max_nfev).x
    for f, p in zip(free, x[:6 * nf].reshape(-1, 6)):
        cams[f] = (cv2.Rodrigues(p[:3])[0], p[3:].copy())
    for t, X in zip(tids, x[6 * nf:].reshape(-1, 3)):
        pts[t] = X


def _projection(cam, K):
    return K @ np.hstack([cam[0], cam[1].reshape(3, 1)])


def _triangulate_new(f, cams, pts, res, K, by_frame, max_err, min_angle):
    """Triangulate tracks seen in the new frame f that have no 3D point yet.

    Pairs f with the earliest registered frame of the track (widest baseline).
    Keeps points in front of both cameras, reprojecting within max_err px and
    seen under at least min_angle deg of parallax (tiny parallax = wild depth).
    """
    groups: dict[int, list[int]] = {}
    for tid in by_frame[f]:
        if tid in pts:
            continue
        g = next((g for g in sorted(res.tracks[tid]) if g != f and g in cams), None)
        if g is not None:
            groups.setdefault(g, []).append(tid)
    for g, tids in groups.items():
        pg = np.array([res.feats[g].pts[res.tracks[t][g]] for t in tids], np.float64)
        pf = np.array([res.feats[f].pts[res.tracks[t][f]] for t in tids], np.float64)
        Xh = cv2.triangulatePoints(_projection(cams[g], K), _projection(cams[f], K), pg.T, pf.T)
        X = (Xh[:3] / Xh[3]).T
        ok = np.isfinite(X).all(1)
        rays = []
        for cam, p in ((cams[g], pg), (cams[f], pf)):
            Xc = X @ cam[0].T + cam[1]
            ok &= Xc[:, 2] > 0
            ok &= np.linalg.norm(reproject(X, cam[0], cam[1], K) - p, axis=1) < max_err
            rays.append(X + cam[0].T @ cam[1])  # X - camera centre
        cos = (rays[0] * rays[1]).sum(1) / (np.linalg.norm(rays[0], axis=1) * np.linalg.norm(rays[1], axis=1))
        ok &= cos < np.cos(np.radians(min_angle))
        for t, X_, keep in zip(tids, X, ok):
            if keep:
                pts[t] = X_


def init_pair(res, K, by_frame, min_rot=3.0, min_common=50):
    """First frame pair (i, j) sharing >= min_common tracks and rotating >= min_rot deg.

    Consecutive pairs can be too close (b barely moves at the start: 0.1 deg), and
    triangulating from a tiny baseline gives garbage depth, so look further ahead.
    Returns (i, j, R, t) or None.
    """
    n = len(res.feats)
    for i in range(n - 1):
        for j in range(i + 1, n):
            tids = [t for t in by_frame.get(i, []) if j in res.tracks[t]]
            if len(tids) < min_common:
                break
            p1 = np.array([res.feats[i].pts[res.tracks[t][i]] for t in tids], np.float64)
            p2 = np.array([res.feats[j].pts[res.tracks[t][j]] for t in tids], np.float64)
            E, m = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
            if E is None:
                continue
            n_in, R, t, _ = cv2.recoverPose(E[:3], p1, p2, K, mask=m)
            if n_in >= min_common and np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])) >= min_rot:
                return i, j, R, t.ravel()
    return None


def reconstruct(res, K, ba_every=5, ba_window=8, max_err=4.0, min_angle=1.0):
    """Step 5: one camera pose per frame and one 3D point per track, in a single world frame.

    1. World = camera i of the initial pair (init_pair), scale = that pair's |t| = 1.
    2. Every later frame: PnP (RANSAC) on its tracks that already have a 3D point,
       so its pose lands in the same frame AND scale; then triangulate new tracks.
    3. Every `ba_every` frames, bundle-adjust the last `ba_window` poses (+ their points).
    4. Finally bundle-adjust everything and drop points that still reproject badly.

    Frames before the initial pair, or with < 6 PnP inliers, stay unregistered.
    Returns (cams {frame: (R, t)} with x_cam = R X + t, pts {track_id: X}).
    """
    by_frame: dict[int, list[int]] = {}
    for tid, tr in enumerate(res.tracks):
        for f in tr:
            by_frame.setdefault(f, []).append(tid)

    init = init_pair(res, K, by_frame)
    if init is None:
        return {}, {}
    i0, j0, R, t = init
    cams = {i0: (np.eye(3), np.zeros(3)), j0: (R, t)}
    pts: dict[int, np.ndarray] = {}
    _triangulate_new(j0, cams, pts, res, K, by_frame, max_err, min_angle)

    for f in range(i0 + 1, len(res.feats)):
        if f in cams:
            continue
        tids = [t for t in by_frame.get(f, []) if t in pts]
        if len(tids) < 6:
            continue
        obj = np.array([pts[t] for t in tids], np.float64)
        img = np.array([res.feats[f].pts[res.tracks[t][f]] for t in tids], np.float64)
        R0, t0 = cams[max(g for g in cams if g < f)]  # nearest earlier pose as the guess
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, K, None, rvec=cv2.Rodrigues(R0)[0], tvec=t0.reshape(3, 1).copy(),
            useExtrinsicGuess=True, reprojectionError=max_err, iterationsCount=200)
        if not ok or inl is None or len(inl) < 6:
            continue
        cams[f] = (cv2.Rodrigues(rvec)[0], tvec.ravel())
        _triangulate_new(f, cams, pts, res, K, by_frame, max_err, min_angle)
        if len(cams) % ba_every == 0:
            bundle_adjust(cams, pts, res, K, free=sorted(cams)[2:][-ba_window:], max_nfev=15)

    # Global pass: frame i0 is the origin; i0+1 is freed so the noisy initial E gets
    # corrected too (scale then floats slightly, which is fine, it is arbitrary anyway).
    bundle_adjust(cams, pts, res, K, free=sorted(cams)[1:])
    err = point_errors(cams, pts, res, K)
    return cams, {t: X for t, X in pts.items() if err[t] < max_err}


def point_errors(cams, pts, res, K):
    """Mean reprojection error (px) of each 3D point over every registered frame that sees it."""
    out = {}
    for tid, X in pts.items():
        e = [np.linalg.norm(reproject(X[None], *cams[f], K)[0] - res.feats[f].pts[k])
             for f, k in res.tracks[tid].items() if f in cams]
        out[tid] = float(np.mean(e))
    return out


def write_ply(path, X, colours):
    """ASCII point cloud. X (N,3) float, colours (N,3) uint8 RGB."""
    with open(path, 'w') as fh:
        fh.write(f'ply\nformat ascii 1.0\nelement vertex {len(X)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for (x, y, z), (r, g, b) in zip(X, colours):
            fh.write(f'{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n')


def track_colours(pts, res, frames):
    """RGB colour of each 3D point, sampled at its first observation."""
    out = []
    for tid in pts:
        f, k = min(res.tracks[tid].items())
        x, y = np.clip(res.feats[f].pts[k].astype(int), 0, [frames[f].shape[1] - 1, frames[f].shape[0] - 1])
        out.append(frames[f][y, x, ::-1])
    return np.array(out, np.uint8).reshape(-1, 3)


if __name__ == '__main__':
    # Self-check: synthetic scene with known motion, recovered R and t direction must match
    rng = np.random.default_rng(0)
    K = np.array([[600., 0, 180], [0, 600, 320], [0, 0, 1]])
    X = rng.uniform([-1, -1, 4], [1, 1, 6], (200, 3))
    R_true, _ = cv2.Rodrigues(np.array([0, 0.05, 0.]))
    t_true = np.array([[0.2], [0], [0]])
    proj = lambda X, R, t: cv2.projectPoints(X, cv2.Rodrigues(R)[0], t, K, None)[0].reshape(-1, 2)
    p1, p2 = proj(X, np.eye(3), np.zeros((3, 1))), proj(X, R_true, t_true)
    E, m = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
    _, R, t, _ = cv2.recoverPose(E, p1, p2, K, mask=m)
    assert np.allclose(R, R_true, atol=1e-3), R
    assert np.allclose(t, t_true / np.linalg.norm(t_true), atol=1e-3), t
    # Step 4: triangulated points equal the true ones up to the scale 1/|t_true|
    tri = triangulate_pairs([{'R': R, 't': t, 'p1': p1, 'p2': p2}], K)[0]
    assert len(tri['X']) == len(X)
    assert np.allclose(tri['X'], X / np.linalg.norm(t_true), rtol=1e-2), tri['X'][:3]
    assert tri['err1'].max() < 0.1 and tri['err2'].max() < 0.1

    # Step 5: camera orbiting the scene, 4-frame tracks + 0.3px noise. Rotations must
    # match the truth and camera spacing must stay at one consistent scale.
    n, step = 16, np.radians(4)
    centre = np.array([0, 0, 5.])
    true = []
    for f in range(n):
        Rf, _ = cv2.Rodrigues(np.array([0, f * step, 0.]))
        true.append((Rf, (centre - Rf @ centre)))  # rotate about the scene centre
    Xw = rng.uniform([-1, -1, 4], [1, 1, 6], (400, 3))
    pix = [proj(Xw, R_, t_.reshape(3, 1)) + rng.normal(0, 0.3, (len(Xw), 2)) for R_, t_ in true]
    feats_ = [features.Features(p.astype(np.float32), np.empty((0, 128), np.float32)) for p in pix]
    tracks = [{f: j for f in range(s, min(s + 4, n))} for j in range(len(Xw)) for s in range(0, n, 2)]
    res = features.FeatureResult(feats_, [], [], tracks)
    cams, pts = reconstruct(res, K)
    assert len(cams) == n, sorted(cams)
    centres = {f: -R_.T @ t_ for f, (R_, t_) in cams.items()}
    true_c = [-R_.T @ t_ for R_, t_ in true]
    s = np.linalg.norm(true_c[1] - true_c[0]) / np.linalg.norm(centres[1] - centres[0])
    for f in range(n):
        assert np.allclose(cams[f][0], true[f][0], atol=1e-2), f
        assert np.allclose(centres[f] * s, true_c[f], atol=0.05 * np.linalg.norm(true_c[-1])), f
    assert np.median(list(point_errors(cams, pts, res, K).values())) < 1.0
    print('ok')
