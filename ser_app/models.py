from django.db import models


class AudioFile(models.Model):
    file = models.FileField(upload_to='audio_files/')
    emotion = models.CharField(max_length=20, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"File: {self.file.name}, Emotion: {self.emotion}"
