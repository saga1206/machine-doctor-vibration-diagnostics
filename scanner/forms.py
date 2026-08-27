"""
Kid explanation: a Django "Form" is a bouncer at the door — it checks
that what came in (the video file) is actually allowed before letting
it into the app (e.g. rejecting a .txt file pretending to be a video).
"""
from django.core.validators import FileExtensionValidator
from django import forms

from .models import Scan

ALLOWED_VIDEO_EXTENSIONS = ["mp4", "webm", "mov", "avi"]


class ScanUploadForm(forms.ModelForm):
    # Free-text option: type a brand new machine name instead of picking
    # an existing one from the dropdown.
    new_machine_name = forms.CharField(
        required=False,
        max_length=100,
        help_text="Leave blank if you picked an existing machine above.",
    )

    class Meta:
        model = Scan
        fields = ["machine", "video_file"]
        widgets = {
            "machine": forms.Select(attrs={"id": "id_machine"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["machine"].required = False
        self.fields["video_file"].validators.append(
            FileExtensionValidator(allowed_extensions=ALLOWED_VIDEO_EXTENSIONS)
        )

    def clean(self):
        cleaned = super().clean()
        machine = cleaned.get("machine")
        new_name = cleaned.get("new_machine_name")
        if not machine and not new_name:
            raise forms.ValidationError(
                "Pick an existing machine or type a name for a new one."
            )
        return cleaned