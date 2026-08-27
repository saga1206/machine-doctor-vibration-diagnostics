"""
Ego-Motion Filtering Module (Step 3)

Kid explanation, longer version:
    Every frame, we watch a few dozen "freckle" points scattered across
    the whole picture. Most freckles sit on the background (walls, table,
    whatever), which doesn't actually move on its own. So if ALL the
    freckles seem to slide together in the same direction, that sliding
    must be the CAMERA moving (or your hand shaking) -- not the machine.

    We fit one simple "camera motion" transform per frame (a small
    shift/rotate/zoom, called an affine transform) that best explains
    how MOST freckles moved. Then, for every freckle, we subtract that
    shared camera motion from its actual movement. Whatever tiny wobble
    is left over is real, local motion -- e.g. the machine vibrating.

Technique used:
    1. cv2.goodFeaturesToTrack   -> pick trackable corner points
    2. cv2.calcOpticalFlowPyrLK  -> follow those points frame-to-frame
       (Lucas-Kanade sparse optical flow)
    3. cv2.estimateAffinePartial2D with RANSAC -> fit the single global
       transform that explains most points' motion (this IS the camera
       motion estimate, because RANSAC naturally ignores the minority
       of points -- e.g. ones sitting on a vibrating machine -- that
       don't agree with the majority)
    4. residual = actual_point_motion - predicted_motion_under_that_transform
       -> this residual is the local vibration signal we actually want

This is intentionally classical computer vision (no neural network),
so it runs comfortably on a CPU with no GPU.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class EgoMotionResult:
    fps: float
    frame_count: int
    # Per-frame estimated camera motion (dx, dy), one pair per frame transition
    camera_motion_x: np.ndarray
    camera_motion_y: np.ndarray
    # Per-frame REAL local vibration signal, after camera motion is removed.
    # This is the single number per frame that Step 5 (FFT) will analyze.
    filtered_signal: np.ndarray
    # For comparison/debugging: what the raw, unfiltered motion looked like
    raw_signal: np.ndarray
    success: bool = True
    error_message: str = ""


def _load_frames_grayscale(video_path, max_frames=None, resize_width=480):
    """
    Kid explanation: opens the video and returns a list of frames as
    grayscale images. We shrink big videos down to resize_width pixels
    wide first, since optical flow is much faster on smaller images and
    we don't need full resolution to see a wobble.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (resize_width, int(h * scale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def filter_camera_motion(video_path, max_frames=None) -> EgoMotionResult:
    """
    Main entry point for Step 3.

    Returns an EgoMotionResult containing, most importantly,
    `filtered_signal`: one number per frame representing how much the
    scene moved AFTER camera shake has been subtracted out. Step 5 will
    run FFT on this signal to find the vibration frequency.
    """
    frames, fps = _load_frames_grayscale(video_path, max_frames=max_frames)

    if len(frames) < 5:
        return EgoMotionResult(
            fps=fps, frame_count=len(frames),
            camera_motion_x=np.array([]), camera_motion_y=np.array([]),
            filtered_signal=np.array([]), raw_signal=np.array([]),
            success=False, error_message="Video too short to analyze (need at least 5 frames).",
        )

    # Parameters for corner detection (the "freckles")
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
    # Parameters for Lucas-Kanade optical flow tracking
    lk_params = dict(
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    prev_gray = frames[0]
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    camera_dx, camera_dy = [], []
    raw_signal, filtered_signal = [], []

    for i in range(1, len(frames)):
        curr_gray = frames[i]

        if prev_pts is None or len(prev_pts) < 10:
            # Lost too many points (e.g. scene changed a lot) -> re-detect
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            if prev_pts is None:
                camera_dx.append(0.0); camera_dy.append(0.0)
                raw_signal.append(0.0); filtered_signal.append(0.0)
                prev_gray = curr_gray
                continue

        curr_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **lk_params
        )

        # Keep only points OpenCV successfully tracked in both frames.
        # cv2 hands points back shaped (N, 1, 2); flatten to (N, 2) so
        # normal 2D array math (hstack, subtraction, etc.) works cleanly.
        good_prev = prev_pts[status.flatten() == 1].reshape(-1, 2)
        good_curr = curr_pts[status.flatten() == 1].reshape(-1, 2)

        if len(good_prev) < 6:
            # Not enough points to trust an affine fit this frame
            camera_dx.append(camera_dx[-1] if camera_dx else 0.0)
            camera_dy.append(camera_dy[-1] if camera_dy else 0.0)
            raw_signal.append(0.0); filtered_signal.append(filtered_signal[-1] if filtered_signal else 0.0)
            prev_gray = curr_gray
            prev_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
            continue

        # --- Step A: raw (unfiltered) motion, just for comparison later ---
        raw_displacement = good_curr - good_prev
        raw_signal.append(float(np.mean(np.linalg.norm(raw_displacement, axis=1))))

        # --- Step B: estimate the single camera-motion transform that
        # best explains MOST points (RANSAC ignores the outlier points,
        # which is exactly the machine's own extra vibration) ---
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=2.0
        )

        if transform is None:
            camera_dx.append(0.0); camera_dy.append(0.0)
            filtered_signal.append(raw_signal[-1])
            prev_gray = curr_gray
            prev_pts = good_curr.reshape(-1, 1, 2)
            continue

        # Camera translation this frame = the transform's shift component
        cam_shift_x, cam_shift_y = transform[0, 2], transform[1, 2]
        camera_dx.append(float(cam_shift_x))
        camera_dy.append(float(cam_shift_y))

        # --- Step C: predict where each point SHOULD be under pure
        # camera motion, then measure the leftover (residual) ---
        ones = np.ones((good_prev.shape[0], 1))
        homo_prev = np.hstack([good_prev, ones])          # Nx3
        predicted_curr = homo_prev @ transform.T           # Nx2, camera-only prediction

        residual = good_curr - predicted_curr               # real local motion
        residual_magnitude = np.linalg.norm(residual, axis=1)

        # The machine is a minority of points that DON'T match camera
        # motion, so we care about the points with the largest residual,
        # not the average (the average is dominated by the well-explained
        # background points, whose residual is ~0 by construction).
        top_k = max(1, int(0.1 * len(residual_magnitude)))  # top 10% of points
        top_indices = np.argsort(residual_magnitude)[-top_k:]

        # IMPORTANT: average the SIGNED vertical residual, not its
        # magnitude/absolute-value. Averaging magnitudes rectifies the
        # wave (like folding a sine wave in half), which DOUBLES its
        # apparent frequency and can alias to a misleading value once
        # FFT'd in Step 5. Keeping the sign preserves the true frequency.
        # (We use the vertical component because most rotating-machine
        # vibration monitoring conventionally looks at one consistent
        # axis; a future improvement could auto-pick the dominant axis
        # per point cluster via PCA instead of assuming "vertical".)
        signed_vertical_residual = residual[top_indices, 1]
        filtered_signal.append(float(np.mean(signed_vertical_residual)))

        prev_gray = curr_gray
        prev_pts = good_curr.reshape(-1, 1, 2)

    return EgoMotionResult(
        fps=fps,
        frame_count=len(frames),
        camera_motion_x=np.array(camera_dx),
        camera_motion_y=np.array(camera_dy),
        filtered_signal=np.array(filtered_signal),
        raw_signal=np.array(raw_signal),
        success=True,
    )