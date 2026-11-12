import numpy as np

from mypylibs.sfm import load_camera, read_frames, relative_poses, triangulate_pairs, undistort

ROOT_IN = 'in'
ROOT_OUT = 'out'
# Use every STEP-th frame: at 15fps consecutive frames of a are ~1 deg apart,
# too small a baseline for a stable E. ponytail: move to in/[a,b].yaml if more params appear
STEP = {'a': 5, 'b': 1}
# No background filtering for b: the rotating checkerboard outvotes the static background in
# RANSAC (median 2.8 deg/pair, fits ~360 deg); dropping matches that move <2px inflated it to 4.7.


def main(name: str):
    K, D = load_camera(ROOT_IN)
    frames = undistort(read_frames(f'{ROOT_IN}/{name}.mp4')[::STEP[name]], K, D)
    poses, feats = relative_poses(frames, K)

    ok = [p for p in poses if p is not None]
    angles = [np.degrees(np.arccos(np.clip((np.trace(p['R']) - 1) / 2, -1, 1))) for p in ok]
    print(f'{name}: {len(frames)} frames, {len(ok)}/{len(poses)} pairs posed, '
          f'median inliers {int(np.median([len(p["p1"]) for p in ok]))}, '
          f'total rotation {sum(angles):.1f} deg')

    tri = [p for p in triangulate_pairs(poses, K) if p is not None]
    err = np.concatenate([np.r_[p['err1'], p['err2']] for p in tri])
    print(f'{name}: {sum(len(p["X"]) for p in tri)} points triangulated, '
          f'reprojection error median {np.median(err):.2f} px, mean {err.mean():.2f} px')


if __name__ == '__main__':
    for name in ('a', 'b'):
        main(name)
