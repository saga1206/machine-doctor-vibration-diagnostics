"""WSGI config for machine_doctor project."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "machine_doctor.settings")

application = get_wsgi_application()