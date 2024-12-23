import librosa
import numpy as np
from keras.models import model_from_json
from keras.layers import Input

# Computed while building model
LABELS_ENCODED = {
    0: 'angry',
    1: 'calm',
    2: 'disgust',
    3: 'fearful',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprised'
}
N_MFCC = 40


def load_model():
    json_file = open('machine_learning_model/SpeechEmotionCNNModel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # Define the input shape
    input_shape = (None, 40, 1)

    # Deserialize the model and specify the input shape
    loaded_model = model_from_json(loaded_model_json, custom_objects={'Input': Input(input_shape)})
    # load weights into new model
    loaded_model.load_weights("machine_learning_model/SpeechEmotionCNNModel.weights.h5")
    print("Loaded model from disk")
    return loaded_model


def cnn_model_prediction(model, file_path_or_buffer):
    # Load the audio file
    audio_signal, sampling_rate = librosa.load(file_path_or_buffer)
    # Perform feature extraction
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sampling_rate, n_mfcc=N_MFCC)
    # Normalize the MFCC features
    normalized_mfcc = np.mean(mfcc.T, axis=0)
    # Add batch and channel dimensions
    input_data = np.expand_dims(normalized_mfcc, axis=0)  # Add batch dimension
    input_data = np.expand_dims(input_data, axis=-1)
    # Predict
    custom_prob = model.predict(input_data)
    custom_predictions = np.argmax(custom_prob, axis=1)
    return [LABELS_ENCODED[custom_predictions[t]] for t in range(len(custom_predictions))]


def analyze_speech_emotion(file):
    model = load_model()
    prediction = cnn_model_prediction(model, file)
    print(f"Successfully Analyzed emotion. {model.__class__.__name__:40} ==> {prediction[0]}")
    return prediction[0]


if __name__ == '__main__':
    m = load_model()
