"""
Motion Magnification Module (Step 4)

Kid explanation:
    Imagine every pixel in the video has a "brightness dial." Between one
    frame and the next, that dial wiggles a tiny bit as the machine
    vibrates -- too tiny to notice with your eyes. This module watches
    each pixel's dial over time and turns the wiggle UP, like turning up
    the volume on a whisper, so a wiggle a human couldn't see becomes
    an obvious pulsing you CAN see.

Technique used (simplified Eulerian Video Magnification):
    1. Convert each frame to grayscale (we only need brightness, not color,
       to see motion -- keeps this fast on a CPU).
    2. Temporally band-pass filter each pixel's brightness over time,
       keeping only the frequency range we care about (e.g. 0.5-25 Hz --
       typical mechanical vibration range, excluding slow lighting
       changes and very fast sensor noise).
    3. Multiply (amplify) that filtered wiggle by a magnification factor.
    4. Add the amplified wiggle back on top of the original frame.

This is the "simplified" version of the real Eulerian Video Magnification
paper (Wu et al. 2012) -- the real paper also does multi-scale spatial
pyramids and color-space tricks for extra quality. We skip those for
speed/simplicity, since a demo just needs the wobble to become visible,
not broadcast-quality output.

Why this runs fine on a CPU:
    It's pure NumPy array math (a per-pixel time-domain filter) --
    no neural network, no GPU needed. For a short 5-15s clip at 480p
    this comfortably finishes in well under our ~30 second budget.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class MagnificationResult:
    output_path: str
    fps: float
    frame_count: int
    magnification_factor: float
    low_freq_hz: float
    high_freq_hz: float
    success: bool = True
    error_message: str = ""


def _bandpass_filter_over_time(pixel_time_series, fps, low_hz, high_hz):
    """
    Kid explanation: this looks at ONE pixel's brightness over the whole
    video (a 1D signal over time) and keeps only the "beat" happening
    between low_hz and high_hz, throwing away anything slower (like a
    light source slowly dimming) or faster (like camera sensor noise).

    Applied independently to every pixel, this isolates exactly the
    vibration-frequency wiggle we want to amplify.
    """
    nyquist = fps / 2.0
    high_hz = min(high_hz, nyquist * 0.99)
    low = low_hz / nyquist
    high = high_hz / nyquist
    b, a = butter(N=2, Wn=[low, high], btype="band")
    return filtfilt(b, a, pixel_time_series, axis=0)


def magnify_video(
    video_path,
    output_path,
    magnification_factor=15.0,
    low_freq_hz=0.5,
    high_freq_hz=25.0,
    resize_width=320,
    max_frames=150,
):
    """
    Main entry point for Step 4.

    Reads `video_path`, amplifies subtle brightness wiggles in the
    low_freq_hz-high_freq_hz range by `magnification_factor`, and writes
    a new video to `output_path` where the vibration is visible.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return MagnificationResult(
            output_path=output_path, fps=0, frame_count=0,
            magnification_factor=magnification_factor,
            low_freq_hz=low_freq_hz, high_freq_hz=high_freq_hz,
            success=False, error_message=f"Could not open video: {video_path}",
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames_gray = []
    frames_color = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            new_w, new_h = resize_width, int(h * scale)
            # H.264 requires both width and height to be EVEN numbers --
            # a resize based on the original video's aspect ratio can
            # easily land on an odd number (e.g. 573), which makes the
            # encoder fail outright. Round down to the nearest even
            # number to guarantee this always works, regardless of the
            # source video's original resolution/aspect ratio.
            new_w -= new_w % 2
            new_h -= new_h % 2
            frame = cv2.resize(frame, (new_w, new_h))
        else:
            # Even un-resized frames need even dimensions for the same
            # reason -- crop a trailing row/column if the source video
            # itself has an odd width or height.
            even_h, even_w = h - (h % 2), w - (w % 2)
            if even_h != h or even_w != w:
                frame = frame[:even_h, :even_w]
        frames_color.append(frame)
        frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
        if len(frames_color) >= max_frames:
            break
    cap.release()

    n_frames = len(frames_gray)
    if n_frames < 10:
        return MagnificationResult(
            output_path=output_path, fps=fps, frame_count=n_frames,
            magnification_factor=magnification_factor,
            low_freq_hz=low_freq_hz, high_freq_hz=high_freq_hz,
            success=False, error_message="Video too short to magnify (need at least 10 frames).",
        )

    stack = np.stack(frames_gray, axis=0)

    h, w = stack.shape[1], stack.shape[2]
    flat = stack.reshape(n_frames, h * w)
    filtered_flat = _bandpass_filter_over_time(flat, fps, low_freq_hz, high_freq_hz)
    filtered = filtered_flat.reshape(n_frames, h, w)

    amplified = filtered * magnification_factor

    # Write output using imageio + the H.264 codec, NOT cv2.VideoWriter's
    # default "mp4v" codec. Kid explanation: mp4v is like writing a letter
    # in a language technically valid but that most people's mailboxes
    # (browsers) can't read -- the file is real and correct, but Chrome/
    # Firefox's <video> tag won't play it. H.264 is the "language" almost
    # every browser understands, so we use that instead for anything the
    # website needs to actually display back to the user.
    import imageio

    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=None,  # avoid imageio silently resizing odd dimensions
    )
    for i in range(n_frames):
        color_frame_bgr = frames_color[i].astype(np.float32)
        boost = amplified[i][:, :, np.newaxis]
        boosted_frame_bgr = np.clip(color_frame_bgr + boost, 0, 255).astype(np.uint8)
        # imageio expects RGB frame order; OpenCV frames are BGR.
        boosted_frame_rgb = cv2.cvtColor(boosted_frame_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(boosted_frame_rgb)
    writer.close()

    return MagnificationResult(
        output_path=str(output_path),
        fps=fps,
        frame_count=n_frames,
        magnification_factor=magnification_factor,
        low_freq_hz=low_freq_hz,
        high_freq_hz=high_freq_hz,
        success=True,
    )