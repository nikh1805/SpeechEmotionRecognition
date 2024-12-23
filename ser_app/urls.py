from django.urls import path
from .views import analyze_speech_emotion_view, audio_file_history_view

urlpatterns = [
    path('', analyze_speech_emotion_view, name='analyze_speech_emotion'),
    path('history/', audio_file_history_view, name='audio_file_history'),
]
