"""Structure from Motion steps.

Step 3: undistort frames, essential matrix E and relative pose (R, t) for every
pair of consecutive frames.
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
    print('ok')
