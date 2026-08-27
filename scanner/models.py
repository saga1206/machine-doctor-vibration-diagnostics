"""
Database models for Machine Doctor.

Kid explanation: a "model" is just a labeled box where we keep
information. Django turns each box into a real database table for us.

We have three boxes:
  1. Machine       -> "which fan/motor is this?"
  2. Scan          -> "one check-up video + its results"
  3. FaultPattern  -> "our little medical textbook of known sicknesses"
"""
from django.db import models


class Machine(models.Model):
    """One physical machine we're keeping track of over time."""

    name = models.CharField(
        max_length=100,
        help_text="A friendly name, e.g. 'Kitchen Exhaust Fan'",
    )
    machine_type = models.CharField(
        max_length=100,
        blank=True,
        help_text="Optional category, e.g. 'fan', 'motor', 'compressor'",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ["-created_at"]


class Scan(models.Model):
    """One check-up: a video of a Machine, plus everything we learned from it."""

    HEALTH_CHOICES = [
        ("healthy", "Healthy"),
        ("watch", "Watch"),
        ("warning", "Warning"),
    ]

    machine = models.ForeignKey(
        Machine, on_delete=models.CASCADE, related_name="scans"
    )
    video_file = models.FileField(upload_to="uploads/videos/")
    created_at = models.DateTimeField(auto_now_add=True)

    # --- filled in AFTER the analysis pipeline runs (steps 3-6) ---
    dominant_frequency_hz = models.FloatField(
        null=True, blank=True,
        help_text="The strongest vibration frequency found by FFT",
    )
    amplitude = models.FloatField(
        null=True, blank=True,
        help_text="How big the vibration is (arbitrary pixel-motion units)",
    )
    camera_shake_removed = models.BooleanField(
        default=False,
        help_text="Whether ego-motion filtering successfully ran",
    )

    # --- filled in by the diagnosis module (step 7) ---
    diagnosis_text = models.TextField(blank=True)
    matched_fault_pattern = models.ForeignKey(
        "FaultPattern", null=True, blank=True, on_delete=models.SET_NULL
    )
    health_status = models.CharField(
        max_length=20, choices=HEALTH_CHOICES, default="healthy"
    )

    # --- generated output files ---
    chart_image = models.ImageField(upload_to="charts/", null=True, blank=True)
    magnified_video = models.FileField(upload_to="magnified/", null=True, blank=True)
    report_pdf = models.FileField(upload_to="reports/", null=True, blank=True)

    # --- processing status, useful once analysis takes a few seconds ---
    is_processed = models.BooleanField(default=False)
    processing_error = models.TextField(blank=True)

    def __str__(self):
        return f"Scan #{self.pk} of {self.machine.name}"

    class Meta:
        ordering = ["-created_at"]


class FaultPattern(models.Model):
    """
    One entry in our 'medical textbook'.

    Example row: "Bearing Wear" happens roughly between 80-200 Hz
    at low amplitude. We match a Scan's measured frequency/amplitude
    against these rows to produce a plain-English diagnosis.
    """

    AMPLITUDE_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
    ]
    HEALTH_CHOICES = [
        ("healthy", "Healthy"),
        ("watch", "Watch"),
        ("warning", "Warning"),
    ]

    name = models.CharField(max_length=100)  # e.g. "Bearing Wear"
    freq_min_hz = models.FloatField()
    freq_max_hz = models.FloatField()
    amplitude_level = models.CharField(max_length=20, choices=AMPLITUDE_CHOICES)
    description = models.TextField(
        help_text="Plain-English description shown to the user"
    )
    recommendation = models.TextField(
        help_text="What the user should do about it"
    )
    health_status = models.CharField(
        max_length=20, choices=HEALTH_CHOICES, default="watch",
        help_text="How serious this fault is, used to color-code the dashboard",
    )

    def __str__(self):
        return self.name