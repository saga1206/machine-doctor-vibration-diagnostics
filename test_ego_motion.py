from scanner.analysis.ego_motion_filter import filter_camera_motion
from scanner.analysis.ego_motion_filter import filter_camera_motion
from scanner.analysis.vibration_fft import analyze_vibration
from scanner.analysis.spatial_3d import estimate_3d_from_video
import numpy as np
import json
from scanner.analysis.diagnosis_rag import diagnose


result = filter_camera_motion("media/uploads/videos/fan_mvt.mp4")  

print("Frames processed:", result.frame_count)
print("Filtered vibration signal:", result.filtered_signal)

signal = result.filtered_signal
fps = result.fps

signal = signal - np.mean(signal)
freqs = np.fft.rfftfreq(len(signal), d=1/fps)
mags = np.abs(np.fft.rfft(signal))
mags[0] = 0  # ignore the "average" bin

peak_freq = freqs[np.argmax(mags)]
print(f"Video FPS: {fps}")
print(f"Dominant vibration frequency: {peak_freq:.2f} Hz")

result = filter_camera_motion("media/uploads/videos/fan_mvt.mp4")
vibration = analyze_vibration(result.filtered_signal, result.fps)

print("Dominant frequency:", vibration.dominant_frequency_hz, "Hz")
print("Amplitude:", vibration.amplitude)
print("Secondary peaks:", vibration.secondary_peaks_hz)

result = estimate_3d_from_video("media/uploads/videos/fan_mvt.mp4")
print("Success:", result.success)
print("Points recovered:", result.n_points)
print("Mean depth (arbitrary units):", result.mean_depth)
print("Depth spread:", result.depth_spread)
print("Error message:", result.error_message)

patterns = json.load(open("fault_knowledge_base.json"))
diagnosis = diagnose(vibration.dominant_frequency_hz, vibration.amplitude, patterns,
                      secondary_peaks_hz=vibration.secondary_peaks_hz)

print("Diagnosis:", diagnosis.matched_pattern_name)
print("Health status:", diagnosis.health_status)
print("Confidence:", diagnosis.confidence)
print("Recommendation:", diagnosis.recommendation)