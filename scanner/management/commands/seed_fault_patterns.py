"""
Kid explanation: this is a little script you run ONCE that reads the
"medical textbook" (fault_knowledge_base.json) off disk and copies
each entry into the real database, so the app can search through it.

Run it with:
    python manage.py seed_fault_patterns
"""
import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from scanner.models import FaultPattern


class Command(BaseCommand):
    help = "Load fault_knowledge_base.json into the FaultPattern table"

    def handle(self, *args, **options):
        json_path = Path(settings.BASE_DIR) / "fault_knowledge_base.json"
        with open(json_path) as f:
            patterns = json.load(f)

        created_count = 0
        for entry in patterns:
            _, created = FaultPattern.objects.update_or_create(
                name=entry["name"],
                defaults={
                    "freq_min_hz": entry["freq_min_hz"],
                    "freq_max_hz": entry["freq_max_hz"],
                    "amplitude_level": entry["amplitude_level"],
                    "description": entry["description"],
                    "recommendation": entry["recommendation"],
                    "health_status": entry.get("health_status", "watch"),
                },
            )
            if created:
                created_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Loaded {len(patterns)} fault patterns ({created_count} new)."
            )
        )