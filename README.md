# CS6524: Computer Vision — Lab 3: Structure from Motion

**Immersive Software Engineering (ISE) — University of Limerick**  

**Instructor:** Juan Albarracín [✉️](mailto:Juan.Albarracin@ul.ie)\
**Teacher Assistant:** Thendral Balakrishnan [✉️](mailto:balakrishnan.thendral@ul.ie)\
**Module Lead:** James Patten [✉️](mailto:James.Patten@ul.ie)

**Deadline:** Sunday, November 15th at Midnight (Autumn Semester Week 3)

---

## 🎯 Objective

Learn how Structure from Motion works and analyse the impact of scene setup and camera parameters in 3D structure estimation.

---

## 📌 Context

Structure from Motion (SfM) can be considered a generalisation of Stereo Vision, by passing from $2$ to $N$ images for 3D estimation. In this laboratory we will focus on how sequential images took by a single camera around a target object, can estimate its structure.
You will shoot two video of an object, for which you will need the intrinsics $K$ of your camera. Hopefully, you already have them from lab2. 

---

## 🔄 Step-by-Step Guide

### 1. Record two videos of an object

- Pick an object with an interesting structure.
- Shoot two specific videos of the object:
    1. Move your camera around the object. Try to cover a complete $360^\circ$ range.
    2. Keep the camera still pointing at the object, and spin the object. Also try to cover the $360^\circ$ range.
- If necessary, pre-process it to keep the videos smaller than 10MB.

### 2. Pick the keypoints you want to appear in the estimated structure

- You may play around a bit with different descriptors like SIFT, SURF and ORB.
- Not all the keypoints will be present in all the frames of the video, so you need to manage the accuracy/density tradeoff:
    - Points that appear in many frames are the most likely to give good predictions.
    - However, such points may be few or none at all, so the structure found can be very sparse.
- The best approach will be to keep as much points as possible on groups of consecutive frames.

### 3. Calculate Essential Matrices

The Essential Matrix $E$ is the transformation between two camera views observing the same scene.

- Grab the intrinsics $K$ and distortion coefficients $D$ from your camera. You might have them already from lab2.
- Undistort the frames of the video.
- For every pair of consecutive frames, find the essential matrix $E$.
- Get the relative position $t$ and rotation $R$.

### 4. Estimate the 3D positions of the points

- Project the points from the first frame assuming the first camera position to be the origin.
- Similarly, project the point of the seocnd frame using the estimated position and rotation of the camera.
- Since both sets of points are expected to represent the same thing in the real world, calulate the *reprojection error* between the 3D estimations of the first and second frame.

### 5. Extend the logic for N frames

By simply doing the above for every two consecutive frames, the reprojection error is expected to accumulate. To solve that, you can either apply least squares to the projections of more than two consecutive points, or every certain number of processed frames, apply a correction envolving five or more frames.

### 6. Dense Structure from Motion

After finding the correct extrinsics of the camera at every frame, map all the pixels from the frame to the 3D space. You can determine yourself how to average the coordinates accross the frames.

### 7. Sensitivity Analysis

Once you find the configuration and algorithm that best estimate the 3D structure of the object, you will have to perform a sensitivity analysis on two variables: focal length and distortion.

- Let $f_x$, $f_y$ be the focal lengths of the camera. Re-run the algorithm multiplying the focal lenghts by $0.8$, $1$ (just the original) and $1.2$.
- Now, remove the undistortion step and run again the experiments above, with the three focal length factors, giving a total of **six** runs.

---

## 📋 Submission Guidelines and Rules

### 1. Repository Setup and Student Identification
- **Forking Requirement:** You must fork the official starter repository into your own GitHub account or team space.
- **Repository Naming Convention:** Rename your forked repository to: `cs6524-lab3-group<XX>`
- **Identification (`TEAM.md`):** Add a `TEAM.md` file in the repository root with the full names, student IDs, and GitHub usernames of all team members before making your final push.

### 2. Code Structure and Execution
- **Main Entry Point:** All execution must be triggered by running `python main.py` in the repository root without any command-line arguments.
- **Custom Modules:** Any additional helper scripts or custom functions must be placed inside the `mypylibs/` directory.
- **Allowed and Prohibited Libraries:** All the libraries present in the docker image are allowed, without any function restriction. The image for this lab has OpenCV.

### 3. Input Specifications (`in/`)
The group only needs two input videos: the spinning camera ($a$) and spinning object ($b$).

- **Naming Convention:** `a.mp4`, `b.mp4`, `K.csv`, `D.csv`, corresponding the two videos asked, and the camera intrinsics and distorion parameters..
- **Optional Configuration:** Videos $a$ and $b$ may require different algorithm configurations, so you can store the params `in/a.yaml` and `in/b.yaml`.
- **Video Rules and Resolution Limits:**
    - Select a static physical scenes containing distinct, well-defined depth layers (e.g., foreground objects at 0.5m, midground elements at 1.5m, and background elements at 3m+). Avoid scenes dominated by textureless, uniform surfaces (like blank walls).
    - Check the video file size: ensure that it does not exceed 5MB.
        - If it does, either configure your camera to take low resolution videos (up to $640 \times 360$ at 30fps), or use `cv2.resize` and video i/o OpenCV functions to create a new video (with less frames if necessary).
        - ⚠️ Make sure the post-calibration videos that you record undergo **exactly the same** pre-processing.
    - You can loosen the rules above a bit, as long as all the required inputs run on Actions smoothly.

### 4. Output Specifications (`out/`)

The output will be point clouds in `.ply` format.
Considering the two scenes, and the sensitivity analysis, you will have to provide **twelve** outputs. The name convention is `[a,b]_ff[u,d].ply` meaning:

- Either using video $a$ or $b$ as input.
- `ff` correspond to the factor so `0.8` becomes `08`, `1` becomes `10`, and `1.2` becomes `12`.
- Either `u` or `d` represents, respectively, undistorted or distorted video.

Example: `a_08d.ply` means video $a$ with focal length multiplied by `0.8`, and no undistortion applied.

### 5. Written Report and Assets (`report.tex` and `assets/latex/`)
- *File Location and Template:* Write your report in `report.tex` in the repository root, using the provided IEEE Transactions LaTeX template.
- **Report Assets (`assets/latex/`):** Place all figures, sub-diagrams, and external files needed for your LaTeX report inside the `assets/latex/` directory to keep the root directory tidy.
- **Length Limit:** Maximum 3 pages (including figures and references).
- **Required Content:**
    - **Methodology:** Concise description of the whole pipeline with particular considerations you find necessary to add.
    - **Discussion and Failure Analysis:** Critical analysis of the results, whether the pipeline succeeded or failed, along with concrete steps for improvement if given more time.
- **Writing Quality and Authenticity:** Be brief, rigorous, and specific. Your analysis must convince the reader that you personally conducted the experiments, understood the mathematical trade-offs, and derived the insights yourself. Avoid pasting generic, boilerplate text from LLMs—reports that read like unverified AI summaries will be flagged and incur heavy grade deductions.

### 6. CI/CD Pipeline, Local Testing and Bonus Point
- **Automated Workflow:** GitHub Actions will execute `main.py` and compile `report.tex` into `report.pdf` upon pushing to `main`.
- **Local Docker Testing:** It is strongly recommended to test your code locally using the provided Docker environment before pushing to ensure pipeline compatibility.
