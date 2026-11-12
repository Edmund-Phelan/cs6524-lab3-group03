"""Shrink raw phone videos to fit the 5MB input limit.

Run once from the repo root:  python mypylibs/preprocess.py
Not called by main.py, the raw videos live in in/raw/ (git-ignored).

Resize to 360x640 (portrait) and keep every 2nd frame (~15fps).
Any calibration video used for K/D must go through the SAME resize,
frame skipping doesn't change K.
"""
import cv2

SIZE = (360, 640)  # (width, height), phone videos are portrait 1080x1920
STEP = 2           # keep every STEP-th frame

VIDEOS = [
    ("in/raw/IMG_2034.MOV", "in/a.mp4"),  # camera moving 360 around object
    ("in/raw/IMG_2035.mov", "in/b.mp4"),  # object spinning, camera still
]


def shrink(src, dst, size=SIZE, step=STEP):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(src)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"),
                          cap.get(cv2.CAP_PROP_FPS) / step, size)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    out.release()


if __name__ == "__main__":
    for src, dst in VIDEOS:
        shrink(src, dst)
        print(f"{src} -> {dst}")
