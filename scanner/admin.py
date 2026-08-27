"""
Kid explanation: this file gives you a free, ready-made control panel
at /admin/ where you can see and edit every Machine, Scan, and
FaultPattern without writing any HTML yourself. Great for testing.
"""
from django.contrib import admin

from .models import FaultPattern, Machine, Scan


@admin.register(Machine)
class MachineAdmin(admin.ModelAdmin):
    list_display = ("name", "machine_type", "created_at")
    search_fields = ("name",)


@admin.register(Scan)
class ScanAdmin(admin.ModelAdmin):
    list_display = (
        "id", "machine", "health_status", "dominant_frequency_hz",
        "amplitude", "is_processed", "created_at",
    )
    list_filter = ("health_status", "is_processed")


@admin.register(FaultPattern)
class FaultPatternAdmin(admin.ModelAdmin):
    list_display = ("name", "freq_min_hz", "freq_max_hz", "amplitude_level")