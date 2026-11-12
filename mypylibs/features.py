"""Step 2: SIFT keypoints, consecutive-frame matching and track building.

A keypoint is identified by (frame_index, keypoint_index). A track is the same
physical point followed across consecutive frames, stored as {frame: keypoint}.
"""

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

DEFAULTS = {
    "n_features": 2000,    # keep the N strongest SIFT keypoints per frame
    "contrast": 0.04,      # SIFT contrastThreshold; lower -> more keypoints
    "ratio": 0.75,         # Lowe's ratio test
    "ransac_thresh": 1.0,  # px; inlier threshold for the essential-matrix RANSAC
    "min_track_len": 2,    # tracks seen in fewer frames are discarded
    "min_motion": 0.0,     # px; drop matches moving less (static background when the object spins)
}


@dataclass
class Features:
    pts: np.ndarray   # (N, 2) float32 pixel coordinates (x, y)
    desc: np.ndarray  # (N, 128) float32 SIFT descriptors


@dataclass
class FeatureResult:
    feats: list[Features]                # one per frame
    pair_matches: list[np.ndarray]       # [i]: (M, 2) inlier matches frame i -> i+1
    essentials: list[np.ndarray | None]  # [i]: E between frame i and i+1, or None
    tracks: list[dict[int, int]]         # {frame_index: keypoint_index}


def detect(frames, n_features=2000, contrast=0.04) -> list[Features]:
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast)
    out = []
    for frame in frames:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
        if desc is None:
            desc = np.empty((0, 128), np.float32)
        out.append(Features(pts, desc))
    return out


_matcher = cv2.BFMatcher(cv2.NORM_L2)


def ratio_match(f1: Features, f2: Features, ratio=0.75) -> np.ndarray:
    """One-to-one matches passing Lowe's ratio test, as (M, 2) int array (idx f1, idx f2)."""
    if len(f1.desc) < 2 or len(f2.desc) < 2:
        return np.empty((0, 2), int)
    best: dict[int, tuple[float, int]] = {}  # idx in f2 -> (distance, idx in f1)
    for pair in _matcher.knnMatch(f1.desc, f2.desc, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            # Several f1 points may pick the same f2 point: keep only the closest.
            if m.trainIdx not in best or m.distance < best[m.trainIdx][0]:
                best[m.trainIdx] = (m.distance, m.queryIdx)
    return np.array([(q, t) for t, (_, q) in best.items()], dtype=int).reshape(-1, 2)


def geometric_filter(f1: Features, f2: Features, matches: np.ndarray, K, thresh=1.0):
    """Keep matches consistent with one camera motion. Returns (inlier matches, E or None)."""
    if len(matches) < 8:
        return matches[:0], None
    p1 = f1.pts[matches[:, 0]]
    p2 = f2.pts[matches[:, 1]]
    E, inl = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=thresh)
    if E is None or inl is None:
        return matches[:0], None
    return matches[inl.ravel().astype(bool)], E[:3]  # E may hold several stacked solutions


def build_tracks(pair_matches: list[np.ndarray], min_len=2) -> list[dict[int, int]]:
    """Chain consecutive-frame matches into tracks {frame_index: keypoint_index}."""
    tracks: list[dict[int, int]] = []
    owner: dict[tuple[int, int], int] = {}  # (frame, keypoint) -> track id
    for i, matches in enumerate(pair_matches):
        for q, t in matches:
            q, t = int(q), int(t)
            tid = owner.get((i, q))
            if tid is None:
                tid = len(tracks)
                tracks.append({i: q})
                owner[(i, q)] = tid
            tracks[tid][i + 1] = t
            owner[(i + 1, t)] = tid
    return [tr for tr in tracks if len(tr) >= min_len]


def run(frames, K, cfg: dict | None = None) -> FeatureResult:
    """Full step 2 on a list of (already undistorted or not) frames."""
    c = {**DEFAULTS, **(cfg or {})}
    K = np.asarray(K, dtype=np.float64)
    feats = detect(frames, c["n_features"], c["contrast"])

    pair_matches, essentials = [], []
    for f1, f2 in pairwise(feats):
        m = ratio_match(f1, f2, c["ratio"])
        # Static points satisfy x2^T E x1 = 0 for any E with R ~ I, so with a still camera
        # they outvote the spinning object in RANSAC. Remove them first.
        m = m[np.linalg.norm(f2.pts[m[:, 1]] - f1.pts[m[:, 0]], axis=1) >= c["min_motion"]]
        m, E = geometric_filter(f1, f2, m, K, c["ransac_thresh"])
        pair_matches.append(m)
        essentials.append(E)

    tracks = build_tracks(pair_matches, c["min_track_len"])
    return FeatureResult(feats, pair_matches, essentials, tracks)
