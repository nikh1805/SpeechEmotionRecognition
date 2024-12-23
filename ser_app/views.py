from django.shortcuts import render, redirect
from .forms import AudioFileForm

from machine_learning_model.run_speech_emotion_model import analyze_speech_emotion
from .models import AudioFile


def analyze_speech_emotion_view(request):
    if request.method == 'POST':
        form = AudioFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file
            audio_file = form.save(commit=False)
            audio_file.save()
            # Analyze Speech Emotion
            audio_file.emotion = analyze_speech_emotion(audio_file.file)
            audio_file.save()
            return render(request, 'upload_audio.html',
                          {'form': form, 'emotion': audio_file.emotion,
                           'audio_file_url': audio_file.file.url})
    else:
        form = AudioFileForm()
    return render(request, 'upload_audio.html', {'form': form, 'emotion': None})


def audio_file_history_view(request):
    audio_file_history = AudioFile.objects.all()
    return render(request, 'history.html', {'audio_file_history': audio_file_history})
