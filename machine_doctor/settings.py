"""
Django settings for machine_doctor project.

Kid explanation: this file is the RULEBOOK for the whole app.
It says things like "where do uploaded videos go?" and
"what apps are turned on?".
"""
from pathlib import Path

# BASE_DIR = the machine_doctor/ folder itself
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Security (fine for local demo, NOT for putting on the real internet) ---
SECRET_KEY = "dev-only-secret-key-change-me-before-deploying"
DEBUG = True
ALLOWED_HOSTS = ["*"]  # local demo only

# --- Apps that are "turned on" ---
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "scanner",  # our one app that does everything
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "machine_doctor.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],  # project-wide templates (optional)
        "APP_DIRS": True,  # also looks inside scanner/templates/
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "machine_doctor.wsgi.application"

# --- Database: SQLite, a single file, zero setup ---
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = []  # no login system for this demo, so skip these

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"

# --- Media files: uploaded videos, generated charts, generated PDFs ---
# Kid explanation: MEDIA_ROOT is the physical shelf on disk where files live.
# MEDIA_URL is the "address" a web browser uses to fetch them.
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- File upload size limit (videos can be a few MB) ---
DATA_UPLOAD_MAX_MEMORY_SIZE = 200 * 1024 * 1024  # 200 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 200 * 1024 * 1024