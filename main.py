import numpy as np

from mypylibs.sfm import load_camera, read_frames, relative_poses, undistort

ROOT_IN = 'in'
ROOT_OUT = 'out'
# Use every STEP-th frame: at 15fps consecutive frames of a are ~1 deg apart,
# too small a baseline for a stable E. ponytail: move to in/[a,b].yaml if more params appear
STEP = {'a': 5, 'b': 1}
# Step-2 settings (see features.DEFAULTS). min_disp drops static background matches in b,
# but the rotating checkerboard already outvotes the background in RANSAC: min_disp=0 gives
# a median 2.8 deg/pair (fits ~360 deg over 124 pairs), min_disp=2 inflates it to 4.7 deg.
FEATURES = {'a': {}, 'b': {}}


def main(name: str):
    K, D = load_camera(ROOT_IN)
    frames = undistort(read_frames(f'{ROOT_IN}/{name}.mp4')[::STEP[name]], K, D)
    poses, feats = relative_poses(frames, K, FEATURES[name])

    ok = [p for p in poses if p is not None]
    angles = [np.degrees(np.arccos(np.clip((np.trace(p['R']) - 1) / 2, -1, 1))) for p in ok]
    print(f'{name}: {len(frames)} frames, {len(ok)}/{len(poses)} pairs posed, '
          f'median inliers {int(np.median([len(p["p1"]) for p in ok]))}, '
          f'total rotation {sum(angles):.1f} deg')


if __name__ == '__main__':
    for name in ('a', 'b'):
        main(name)
