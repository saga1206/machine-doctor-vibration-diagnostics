"""
Full Analysis Pipeline (Step 8)

Kid explanation:
    Every module we built (Steps 3-7) has been sitting on a workbench,
    tested one at a time by hand. This file is the CONVEYOR BELT: the
    moment a video is uploaded, it automatically carries that video
    through every station in order:

        video -> [ego-motion filter] -> [FFT] -> [3D depth, optional]
              -> [AI diagnosis] -> [chart] -> saved results

    and writes the results directly onto the Scan record, so the
    website can just display scan.diagnosis_text, scan.health_status,
    etc. without knowing or caring how they were computed.

Design choice -- graceful degradation:
    The CORE result (frequency + amplitude + diagnosis) is required for
    a scan to count as "processed." The 3D depth estimate (Step 6) and
    the magnified video (Step 4) are treated as OPTIONAL extras: if a
    video is too steady for 3D depth (a known, expected limitation from
    Step 6), or magnification fails for some reason, we still want the
    user to see their diagnosis rather than a broken page. Any partial
    failures are recorded in `processing_error` as warnings, not fatal
    errors.
"""
import io
import os
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")  # headless backend -- no display available on a server
import matplotlib.pyplot as plt

from django.core.files import File
from django.core.files.base import ContentFile

from .diagnosis_rag import diagnose
from .ego_motion_filter import filter_camera_motion
from .generation import generate_diagnosis_narrative
from .motion_magnify import magnify_video
from .report_pdf import generate_pdf_bytes
from .spatial_3d import estimate_3d_from_video
from .vibration_fft import analyze_vibration


def _generate_spectrum_chart(vibration_features, scan_id):
    """
    Kid explanation: draws a simple picture of the FFT results -- a
    graph showing which frequencies were found and how strong each one
    was, with the winning (dominant) frequency clearly marked. This is
    the chart shown on the scan result page and later embedded in the
    PDF report (Step 10).

    Returns a Django ContentFile ready to attach to an ImageField.
    """
    freqs = vibration_features.spectrum_freqs
    mags = vibration_features.spectrum_magnitudes

    # Only plot up to ~50 Hz -- typical small-machine vibration range --
    # so the interesting part of the chart isn't squished by high-frequency
    # noise bins nobody cares about.
    plot_mask = freqs <= min(50, freqs.max() if len(freqs) else 50)

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
    ax.plot(freqs[plot_mask], mags[plot_mask], color="#2b6cb0", linewidth=1.5)
    ax.axvline(
        vibration_features.dominant_frequency_hz, color="#e53e3e",
        linestyle="--", linewidth=1,
        label=f"Dominant: {vibration_features.dominant_frequency_hz} Hz",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Scan #{scan_id} — Vibration Frequency Spectrum")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return ContentFile(buf.read(), name=f"scan_{scan_id}_spectrum.png")


def run_full_pipeline(scan):
    """
    Main entry point for Step 8.

    Takes a saved Scan (with a video_file already on disk) and runs it
    through Steps 3-7, writing every result directly onto the `scan`
    object and saving it. Designed to be called right after a Scan is
    created in the upload view.
    """
    warnings = []
    video_path = scan.video_file.path

    try:
        # --- Step 3: ego-motion filtering ---
        ego_result = filter_camera_motion(video_path)
        if not ego_result.success:
            scan.is_processed = False
            scan.processing_error = f"Could not analyze video motion: {ego_result.error_message}"
            scan.save()
            return scan

        scan.camera_shake_removed = True

        # --- Step 5: FFT vibration analysis ---
        vibration = analyze_vibration(ego_result.filtered_signal, ego_result.fps)
        if not vibration.success:
            scan.is_processed = False
            scan.processing_error = f"Could not extract a vibration frequency: {vibration.error_message}"
            scan.save()
            return scan

        scan.dominant_frequency_hz = vibration.dominant_frequency_hz
        scan.amplitude = vibration.amplitude

        # --- Step 6: 3D spatial registration (OPTIONAL -- don't fail the
        # whole scan if the camera was too steady for depth estimation;
        # this is an expected, known limitation, not a real error) ---
        try:
            spatial = estimate_3d_from_video(video_path)
            if not spatial.success:
                warnings.append(f"3D depth unavailable: {spatial.error_message}")
        except Exception as e:
            warnings.append(f"3D depth estimation crashed unexpectedly: {e}")

        # --- Step 7: AI diagnosis ---
        from scanner.models import FaultPattern

        patterns = list(FaultPattern.objects.values(
            "name", "freq_min_hz", "freq_max_hz", "amplitude_level",
            "description", "recommendation", "health_status",
        ))
        diagnosis = diagnose(
            vibration.dominant_frequency_hz, vibration.amplitude, patterns,
            secondary_peaks_hz=vibration.secondary_peaks_hz,
        )

        if diagnosis.success:
            # RAG's "Generation" step: hand the RETRIEVED fault pattern
            # + measured numbers to a real LLM to write the final
            # explanation, instead of just pasting the retrieved
            # description into a template.
            generation = generate_diagnosis_narrative(
                frequency_hz=vibration.dominant_frequency_hz,
                amplitude=vibration.amplitude,
                matched_pattern_name=diagnosis.matched_pattern_name,
                description=diagnosis.description,
                recommendation=diagnosis.recommendation,
                confidence=diagnosis.confidence,
                secondary_peaks_hz=vibration.secondary_peaks_hz,
            )
            scan.diagnosis_text = generation.text
            if not generation.used_llm:
                warnings.append(f"AI narrative generation: {generation.error_message}")

            scan.health_status = diagnosis.health_status
            scan.matched_fault_pattern = FaultPattern.objects.filter(
                name=diagnosis.matched_pattern_name
            ).first()
        else:
            warnings.append(f"Diagnosis unavailable: {diagnosis.error_message}")

        # --- Chart generation ---
        try:
            chart_file = _generate_spectrum_chart(vibration, scan.pk)
            scan.chart_image.save(chart_file.name, chart_file, save=False)
        except Exception as e:
            warnings.append(f"Chart generation failed: {e}")

        # --- Step 4: motion magnification (OPTIONAL, purely for visual
        # demo value -- a failure here shouldn't hide the diagnosis) ---
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            mag_result = magnify_video(video_path, tmp_path, magnification_factor=15.0)
            if mag_result.success:
                with open(tmp_path, "rb") as f:
                    scan.magnified_video.save(
                        f"scan_{scan.pk}_magnified.mp4", File(f), save=False
                    )
            else:
                warnings.append(f"Motion magnification unavailable: {mag_result.error_message}")
            os.remove(tmp_path)
        except Exception as e:
            warnings.append(f"Motion magnification failed: {e}")

        # --- Step 10: PDF report (generated LAST, since it needs the
        # chart image and diagnosis text from the steps above) ---
        try:
            scan_data = {
                "machine_name": scan.machine.name,
                "created_at": scan.created_at.strftime("%b %d, %Y, %H:%M") if scan.created_at else "",
                "health_status": scan.health_status,
                "dominant_frequency_hz": scan.dominant_frequency_hz,
                "amplitude": scan.amplitude,
                "diagnosis_text": scan.diagnosis_text,
                "chart_image_path": scan.chart_image.path if scan.chart_image else None,
            }
            pdf_bytes = generate_pdf_bytes(scan_data)
            scan.report_pdf.save(
                f"scan_{scan.pk}_report.pdf", ContentFile(pdf_bytes), save=False
            )
        except Exception as e:
            warnings.append(f"PDF report generation failed: {e}")

        scan.is_processed = True
        scan.processing_error = " | ".join(warnings)  # non-fatal notes, if any
        scan.save()
        return scan

    except Exception as e:
        # Catch-all: something unexpected happened outside the per-step
        # try/excepts above. Record it clearly instead of a raw 500 error.
        scan.is_processed = False
        scan.processing_error = f"Unexpected error during analysis: {e}\n{traceback.format_exc()}"
        scan.save()
        return scan