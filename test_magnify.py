from scanner.analysis.motion_magnify import magnify_video

result = magnify_video(
    "media/uploads/videos/fan_mvt.mp4",
    "magnified_output.mp4",
    magnification_factor=15.0,
)
print("Success:", result.success)
print("Saved to:", result.output_path)