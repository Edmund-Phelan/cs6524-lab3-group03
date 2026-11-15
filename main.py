import csv
import time

import cv2
import numpy as np

from mypylibs.sfm import (load_camera, point_errors, read_frames, reconstruct, relative_poses,
                          track_colours, triangulate_pairs, undistort, write_ply)
from mypylibs.visualise import save_video

ROOT_IN = 'in'
ROOT_OUT = 'out'
# Use every STEP-th frame: at 15fps consecutive frames of a are ~1 deg apart,
# too small a baseline for a stable E. ponytail: move to in/[a,b].yaml if more params appear
STEP = {'a': 5, 'b': 1}
# b: camera still, object spins. ~70% of matches are static background (<1px), which fits
# "camera didn't move" and dragged step 5 to 103 deg total. Dropping them -> 273 deg (truth ~360).
# 1.25px+ breaks the chain around frame 37 (too few moving matches left there).
CFG = {'a': {}, 'b': {'min_motion': 1.0}}
# Step 7: focal lengths fx, fy scaled by these factors, with and without undistortion.
# The scaled K is used everywhere, undistortion included, i.e. a consistently wrong calibration.
FACTORS = (0.8, 1.0, 1.2)


def run(name, raw, K0, D, factor, undist):
    """Steps 2-5 for one video and one calibration setting. Writes the sparse PLY, returns metrics."""
    t0 = time.time()
    K = K0.copy()
    K[0, 0] *= factor
    K[1, 1] *= factor
    frames = undistort(raw, K, D) if undist else raw
    tag = f'{name}_{round(factor * 10):02d}{"u" if undist else "d"}'

    # Steps 2-4: features, pairwise poses, pairwise triangulation
    poses, feats = relative_poses(frames, K, CFG[name])
    tri = [p for p in triangulate_pairs(poses, K) if p is not None]
    err4 = np.concatenate([np.r_[p['err1'], p['err2']] for p in tri])

    # Step 5: all frames in one world frame and scale
    cams, pts = reconstruct(feats, K)
    err5 = np.array(list(point_errors(cams, pts, feats, K).values()))
    fs = sorted(cams)
    rot = sum(np.degrees(np.linalg.norm(cv2.Rodrigues(cams[b][0] @ cams[a][0].T)[0])) for a, b in zip(fs, fs[1:]))
    write_ply(f'{ROOT_OUT}/{tag}.ply', np.array(list(pts.values())).reshape(-1, 3),
              track_colours(pts, feats, frames))

    if factor == 1.0 and undist:  # reference run: also keep the keypoint/track overlay video
        save_video(frames, feats, f'{ROOT_OUT}/{name}_overlay.mp4')

    m = {
        'run': tag,
        'pairs_posed': f'{sum(p is not None for p in poses)}/{len(poses)}',
        'step4_err_px': round(float(np.median(err4)), 3),
        'registered': f'{len(cams)}/{len(frames)}',
        'sparse_pts': len(pts),
        'step5_err_px': round(float(np.median(err5)), 3),
        'rotation_deg': round(float(rot), 1),
        'seconds': round(time.time() - t0, 1),
    }
    print(', '.join(f'{k}={v}' for k, v in m.items()), flush=True)
    return m


if __name__ == '__main__':
    K0, D = load_camera(ROOT_IN)
    rows = []
    for name in ('a', 'b'):
        raw = read_frames(f'{ROOT_IN}/{name}.mp4')[::STEP[name]]
        for undist in (True, False):
            for factor in FACTORS:
                rows.append(run(name, raw, K0, D, factor, undist))
    with open(f'{ROOT_OUT}/metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
