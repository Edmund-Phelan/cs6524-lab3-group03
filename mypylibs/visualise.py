"""Step 2 visualisations: keypoints, matches and tracks drawn on the frames.

OpenCV only (no matplotlib), so it runs in the CI docker image as-is.
Every function returns a BGR image; save it with cv2.imwrite.
"""
import cv2
import numpy as np

from mypylibs.features import FeatureResult, Features


def _colour(i):
    """Stable, distinct-ish BGR colour per id."""
    hue = (i * 47) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def draw_keypoints(frame, feats: Features, radius=2):
    """All SIFT keypoints of one frame as green dots."""
    img = frame.copy()
    for x, y in feats.pts:
        cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(img, f'{len(feats.pts)} kps', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def draw_matches(frame1, frame2, f1: Features, f2: Features, matches: np.ndarray):
    """Frames side by side, one line per inlier match (i -> i+1)."""
    img = np.hstack([frame1, frame2])
    w = frame1.shape[1]
    for k, (q, t) in enumerate(matches):
        (x1, y1), (x2, y2) = f1.pts[q], f2.pts[t]
        cv2.line(img, (int(x1), int(y1)), (int(x2) + w, int(y2)), _colour(k), 1, cv2.LINE_AA)
    cv2.putText(img, f'{len(matches)} inliers', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def draw_tracks(frame, res: FeatureResult, frame_idx: int, min_len=2):
    """Trail of every track alive at frame_idx, from its first frame up to frame_idx."""
    img = frame.copy()
    n = 0
    for tid, tr in enumerate(res.tracks):
        if frame_idx not in tr:
            continue
        pts = [res.feats[f].pts[k] for f, k in sorted(tr.items()) if f <= frame_idx]
        if len(pts) < min_len:
            continue
        n += 1
        poly = np.int32(pts).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], False, _colour(tid), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(poly[-1, 0]), 2, _colour(tid), -1, cv2.LINE_AA)
    cv2.putText(img, f'{n} tracks', (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def save_video(frames, res: FeatureResult, path: str, fps=10):
    """Whole sequence as an mp4: keypoints (green dots) + track trails on every frame."""
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for i, frame in enumerate(frames):
        out.write(draw_tracks(draw_keypoints(frame, res.feats[i]), res, i))
    out.release()


def track_length_hist(res: FeatureResult, width=400, height=200):
    """Bar chart of track lengths (frames per track): the accuracy/density tradeoff at a glance."""
    lengths = np.bincount([len(t) for t in res.tracks]) if res.tracks else np.zeros(1, int)
    img = np.full((height, width, 3), 255, np.uint8)
    bw = width / max(len(lengths), 1)
    top = max(lengths.max(), 1)
    for L, count in enumerate(lengths):
        h = int((height - 25) * count / top)
        cv2.rectangle(img, (int(L * bw), height - 15 - h), (int((L + 1) * bw) - 1, height - 15), (200, 120, 0), -1)
    cv2.putText(img, f'track length 0..{len(lengths) - 1}, max count {top}', (5, height - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img


def save_step2(frames, res: FeatureResult, prefix: str, frame_idx: int = 0):
    """Write keypoints / matches / tracks / histogram PNGs for one frame, e.g. prefix='out/a'."""
    i = min(frame_idx, len(frames) - 2)
    cv2.imwrite(f'{prefix}_keypoints.png', draw_keypoints(frames[i], res.feats[i]))
    cv2.imwrite(f'{prefix}_matches.png', draw_matches(frames[i], frames[i + 1], res.feats[i],
                                                      res.feats[i + 1], res.pair_matches[i]))
    cv2.imwrite(f'{prefix}_tracks.png', draw_tracks(frames[i + 1], res, i + 1))
    cv2.imwrite(f'{prefix}_track_lengths.png', track_length_hist(res))
