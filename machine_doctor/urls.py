"""
Project-level URL map.

Kid explanation: this is the app's "front door directory" — it says
"if someone knocks on /admin, send them to the admin office; if they
knock on anything else, send them to the scanner app's own directory."
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("scanner.urls")),
]

# Serve uploaded videos / charts / PDFs while DEBUG=True (local dev only)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)