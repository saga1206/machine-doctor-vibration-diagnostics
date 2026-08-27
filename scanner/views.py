"""
Views = the "waiters" of the app. A view takes a request ("I want the
dashboard page"), goes and gets the right information from the models,
and hands back a rendered HTML page.
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.shortcuts import get_object_or_404, redirect, render

from .analysis.trend_prediction import analyze_trend
from .forms import ScanUploadForm
from .models import Machine, Scan


def dashboard(request):
    machines = Machine.objects.prefetch_related("scans").all()
    return render(request, "scanner/dashboard.html", {"machines": machines})


def upload_video(request):
    if request.method == "POST":
        form = ScanUploadForm(request.POST, request.FILES)
        if form.is_valid():
            scan = form.save(commit=False)

            # If the user typed a brand-new machine name instead of
            # picking one from the dropdown, create it now.
            new_name = form.cleaned_data.get("new_machine_name")
            if new_name:
                machine, _ = Machine.objects.get_or_create(name=new_name)
                scan.machine = machine

            scan.save()

            # Run the full analysis pipeline (Steps 3-7) right now,
            # before showing the result page. This is a synchronous call
            # -- the page will wait for it -- which is fine since our
            # CPU-only pipeline finishes in well under our ~30 second
            # budget for a short clip. A production app with heavier
            # analysis or many simultaneous users would instead hand
            # this off to a background task queue (e.g. Celery), but
            # that's unnecessary complexity for this demo's scale.
            from .analysis.pipeline import run_full_pipeline
            run_full_pipeline(scan)

            return redirect("scan_result", pk=scan.pk)
        else:
            # Print the EXACT validation errors to the terminal running
            # `runserver`, so it's obvious what went wrong instead of
            # guessing. Look at your terminal after submitting to see this.
            print("=== UPLOAD FORM VALIDATION FAILED ===")
            print(form.errors.as_text())
            print("======================================")
    else:
        form = ScanUploadForm()

    machines = Machine.objects.all()
    return render(
        request, "scanner/upload.html", {"form": form, "machines": machines}
    )


def scan_result(request, pk):
    scan = get_object_or_404(Scan, pk=pk)
    return render(request, "scanner/scan_result.html", {"scan": scan})


def _generate_trend_chart_data_uri(scans):
    """
    Kid explanation: draws a simple line chart of amplitude over time
    for one machine's scan history, and hands it back as a string the
    HTML template can drop directly into an <img> tag -- no separate
    image file needs to be saved to disk for this, since it's cheap to
    redraw fresh every time the page loads from data already in the
    database.
    """
    dates = [s.created_at.strftime("%m/%d %H:%M") for s in scans]
    amplitudes = [s.amplitude for s in scans]

    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
    ax.plot(dates, amplitudes, marker="o", color="#2b6cb0")
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude Trend Over Time")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def machine_detail(request, pk):
    machine = get_object_or_404(Machine, pk=pk)
    scans = machine.scans.filter(is_processed=True).order_by("created_at")

    trend = analyze_trend([s.amplitude for s in scans])

    chart_data_uri = None
    if scans.count() >= 2:
        chart_data_uri = _generate_trend_chart_data_uri(scans)

    return render(request, "scanner/machine_detail.html", {
        "machine": machine,
        "scans": scans.order_by("-created_at"),  # newest first for display
        "trend": trend,
        "chart_data_uri": chart_data_uri,
    })