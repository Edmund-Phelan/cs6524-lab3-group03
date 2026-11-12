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


def main(name: str):
    K, D = load_camera(ROOT_IN)
    frames = undistort(read_frames(f'{ROOT_IN}/{name}.mp4')[::STEP[name]], K, D)
    poses, feats = relative_poses(frames, K, CFG[name])
    save_video(frames, feats, f'{ROOT_OUT}/{name}_overlay.mp4')

    ok = [p for p in poses if p is not None]
    angles = [np.degrees(np.arccos(np.clip((np.trace(p['R']) - 1) / 2, -1, 1))) for p in ok]
    print(f'{name}: {len(frames)} frames, {len(ok)}/{len(poses)} pairs posed, '
          f'median inliers {int(np.median([len(p["p1"]) for p in ok]))}, '
          f'total rotation {sum(angles):.1f} deg')

    tri = [p for p in triangulate_pairs(poses, K) if p is not None]
    err = np.concatenate([np.r_[p['err1'], p['err2']] for p in tri])
    print(f'{name}: {sum(len(p["X"]) for p in tri)} points triangulated, '
          f'reprojection error median {np.median(err):.2f} px, mean {err.mean():.2f} px')

    # Step 5: all frames in one world frame and scale
    cams, pts = reconstruct(feats, K)
    err = np.array(list(point_errors(cams, pts, feats, K).values()))
    fs = sorted(cams)
    rot = sum(np.degrees(np.linalg.norm(cv2.Rodrigues(cams[b][0] @ cams[a][0].T)[0])) for a, b in zip(fs, fs[1:]))
    print(f'{name}: step 5 registered {len(cams)}/{len(frames)} frames, {len(pts)} points, '
          f'reprojection error median {np.median(err):.2f} px, total rotation {rot:.1f} deg')
    write_ply(f'{ROOT_OUT}/{name}_sparse.ply', np.array(list(pts.values())), track_colours(pts, feats, frames))


if __name__ == '__main__':
    for name in ('a', 'b'):
        main(name)
