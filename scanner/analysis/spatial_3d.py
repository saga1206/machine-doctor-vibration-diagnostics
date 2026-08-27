"""
3D Spatial Registration Module (Step 6)

Kid explanation:
    Close one eye and try to judge how far away your hand is -- hard,
    right? Now move your head a little to the side while looking at it.
    Your brain compares "how much did my hand seem to shift compared to
    the wall behind it?" -- things close by shift a LOT, things far away
    barely shift at all. That comparison is literally how depth
    perception works, and it's exactly what this module does with two
    video frames instead of two eyes.

Technique used (classical two-view Structure-from-Motion, NOT full SLAM):
    1. Track the same corner points across two frames with some time
       gap between them (more gap = more "parallax" = better depth cues,
       like moving your head further for a bigger baseline).
    2. Use the Essential Matrix (cv2.findEssentialMat) to figure out how
       the camera itself moved (rotated/translated) between those two
       frames, using ONLY the 2D point movements -- no special hardware.
    3. Recover the actual rotation+translation (cv2.recoverPose).
    4. Triangulate: once we know how the camera moved AND where each
       point appears in both frames, simple geometry (like GPS
       triangulation) pins down each point's approximate 3D position.

IMPORTANT honest limitation:
    A single uncalibrated camera can only recover 3D structure "up to an
    unknown scale" -- we can tell that point A is twice as far away as
    point B, but not whether that's 2 meters or 20 meters, unless we
    know the real-world size of something in the scene. This is normal
    and expected for monocular (single-camera) 3D estimation; a real
    industrial system would use a known calibration target or two
    physical cameras (stereo) to fix the scale. We keep this lightweight
    on purpose, matching the "no full SLAM" requirement.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SpatialResult:
    points_3d: np.ndarray          # (N, 3) approximate positions, unknown global scale
    rotation: np.ndarray           # (3, 3) camera rotation between the two frames
    translation: np.ndarray        # (3, 1) camera translation direction (unit length, scale unknown)
    mean_depth: float
    depth_spread: float            # how much depth varies across points (std dev)
    n_points: int
    success: bool = True
    error_message: str = ""


def _approximate_intrinsics(width, height):
    """
    Kid explanation: to turn 2D pixel positions into 3D geometry, the
    math needs to know a bit about the camera's lens (the "intrinsic
    matrix" K). We don't have the user's real camera calibration, so we
    use a common, reasonable guess: focal length roughly equal to the
    image width, and the optical center at the middle of the frame.
    This is an approximation -- good enough for relative/demo 3D
    structure, not for measurement-grade accuracy.
    """
    focal_length = float(max(width, height))
    cx, cy = width / 2.0, height / 2.0
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    return K


def estimate_pose_and_structure(pts1, pts2, K):
    """
    Core geometry step, kept separate from video-loading so it can be
    tested directly against synthetic point correspondences with a
    known correct answer.

    pts1, pts2: (N, 2) arrays of matching 2D points in frame 1 and frame 2.
    K: (3, 3) camera intrinsic matrix.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    if len(pts1) < 8:
        return SpatialResult(
            points_3d=np.array([]), rotation=np.eye(3), translation=np.zeros((3, 1)),
            mean_depth=0, depth_spread=0, n_points=0,
            success=False, error_message="Need at least 8 point correspondences for 3D estimation.",
        )

    # Essential matrix: encodes the camera's rotation+translation between
    # the two views, estimated purely from how the points moved.
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return SpatialResult(
            points_3d=np.array([]), rotation=np.eye(3), translation=np.zeros((3, 1)),
            mean_depth=0, depth_spread=0, n_points=0,
            success=False, error_message="Could not estimate camera motion (not enough parallax between frames).",
        )

    # Recover the actual R, t from the Essential matrix (there are 4
    # mathematically possible answers; recoverPose picks the physically
    # sensible one where points end up in FRONT of both camera views).
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    # Keep only the points recoverPose confirmed are consistent (in front
    # of the camera in both views -- the "cheirality" check)
    inlier_idx = pose_mask.flatten() > 0
    pts1_in = pts1[inlier_idx]
    pts2_in = pts2[inlier_idx]

    if len(pts1_in) < 4:
        return SpatialResult(
            points_3d=np.array([]), rotation=R, translation=t,
            mean_depth=0, depth_spread=0, n_points=0,
            success=False, error_message="Too few reliable points after filtering to triangulate.",
        )

    # Camera 1 is treated as the origin looking down +Z; camera 2's pose
    # is the R, t we just recovered relative to it.
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    points_4d = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
    points_3d = (points_4d[:3] / points_4d[3]).T  # convert from homogeneous coords

    # Sanity filter: drop points with negative or absurd depth (numerical
    # junk from near-zero-parallax points), keep the physically sensible ones.
    depths = points_3d[:, 2]
    valid = (depths > 0) & (depths < np.percentile(depths[depths > 0], 99) * 3 if np.any(depths > 0) else depths > 0)
    points_3d_clean = points_3d[valid] if np.any(valid) else points_3d

    if len(points_3d_clean) == 0:
        return SpatialResult(
            points_3d=np.array([]), rotation=R, translation=t,
            mean_depth=0, depth_spread=0, n_points=0,
            success=False, error_message="No points had a valid (positive) depth after triangulation.",
        )

    return SpatialResult(
        points_3d=points_3d_clean,
        rotation=R,
        translation=t,
        mean_depth=float(np.mean(points_3d_clean[:, 2])),
        depth_spread=float(np.std(points_3d_clean[:, 2])),
        n_points=len(points_3d_clean),
        success=True,
    )


def estimate_3d_from_video(video_path, resize_width=480, frame_gap_fraction=0.5):
    """
    Main entry point for Step 6.

    Kid explanation: picks two frames from the video that are reasonably
    far apart in time (more natural hand movement between them = more
    parallax = a better depth estimate), tracks matching points between
    them, and hands off to estimate_pose_and_structure() above.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return SpatialResult(
            points_3d=np.array([]), rotation=np.eye(3), translation=np.zeros((3, 1)),
            mean_depth=0, depth_spread=0, n_points=0,
            success=False, error_message=f"Could not open video: {video_path}",
        )

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (resize_width, int(h * scale)))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(frames) < 5:
        return SpatialResult(
            points_3d=np.array([]), rotation=np.eye(3), translation=np.zeros((3, 1)),
            mean_depth=0, depth_spread=0, n_points=0,
            success=False, error_message="Video too short for 3D estimation.",
        )

    h, w = frames[0].shape
    K = _approximate_intrinsics(w, h)

    feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
    lk_params = dict(
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # A perfectly steady tripod shot (which is actually IDEAL for vibration
    # measurement in Steps 3-5!) gives this module nothing to compare, since
    # 3D depth needs the camera itself to shift position a little, the same
    # way your two eyes need to be in different spots to judge distance.
    # So instead of trying only one frame pair, we try several with
    # progressively more time between them, giving natural hand tremor the
    # best chance to provide enough parallax before we give up.
    candidate_fractions = sorted(set([frame_gap_fraction, 0.3, 0.5, 0.7, 0.95]))
    last_error = "No frame pair had enough camera movement to estimate depth."

    for fraction in candidate_fractions:
        idx2 = min(len(frames) - 1, max(1, int(len(frames) * fraction)))
        frame1, frame2 = frames[0], frames[idx2]

        pts1 = cv2.goodFeaturesToTrack(frame1, mask=None, **feature_params)
        if pts1 is None:
            last_error = "No trackable features found in the first frame."
            continue

        pts2, status, _ = cv2.calcOpticalFlowPyrLK(frame1, frame2, pts1, None, **lk_params)
        good1 = pts1[status.flatten() == 1].reshape(-1, 2)
        good2 = pts2[status.flatten() == 1].reshape(-1, 2)

        result = estimate_pose_and_structure(good1, good2, K)
        if result.success:
            return result
        last_error = result.error_message

    # Every attempt failed -- almost always means the camera genuinely
    # didn't move enough between any two frames for depth to be computable.
    # This is expected physics, not a crash: report it clearly so the rest
    # of the app (Step 8) can show "3D depth unavailable" instead of dying.
    return SpatialResult(
        points_3d=np.array([]), rotation=np.eye(3), translation=np.zeros((3, 1)),
        mean_depth=0, depth_spread=0, n_points=0,
        success=False,
        error_message=(
            "Not enough camera movement (parallax) across the clip to estimate "
            "3D depth. This is normal for a very steady shot -- which is actually "
            "good for the vibration measurement in other steps -- but monocular "
            "depth needs at least a little natural hand movement to work. "
            f"(Last attempt detail: {last_error})"
        ),
    )