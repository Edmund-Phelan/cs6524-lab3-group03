"""Structure from Motion steps.

Step 3: undistort frames, essential matrix E and relative pose (R, t) for every
pair of consecutive frames.
Step 4: triangulate each pair's inlier points into 3D (camera i at the origin)
and measure the reprojection error in both frames.
"""
import cv2
import numpy as np

try:
    from mypylibs import features
except ImportError:  # run directly as python mypylibs/sfm.py
    import features


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
    print('ok')
