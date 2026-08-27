#!/usr/bin/env python
"""Django's command-line utility for administrative tasks.

Kid explanation: this is the "ON switch" for the whole app.
You run `python manage.py runserver` and it wakes everything up.
"""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "machine_doctor.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Did you forget to activate a virtual "
            "environment and run `pip install -r requirements.txt`?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()