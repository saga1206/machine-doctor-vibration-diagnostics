from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload_video, name="upload"),
    path("scan/<int:pk>/", views.scan_result, name="scan_result"),
    path("machine/<int:pk>/", views.machine_detail, name="machine_detail"),
]