# Speech Emotion Recognition
An Emotion Detection system using machine learning models to identify emotions in speech signals, such as calm, happy, sad, and angry, by decoding emotional expressions from audio.

## **Overview**
This project is a Speech Emotion Recognition system that identifies emotions from speech audio files. It has two primary components:
1. **Notebook-based Model Development**:
   - Building and exporting the speech emotion recognition model using the RAVDESS dataset.
2. **Django Web Application**:
   - A user-friendly web interface where users can upload custom audio files to analyze and predict emotions.

----
## **1. Notebook: Model Development**
The notebook (`machine_learning_model/SpeechEmotionDetection.ipynb`) is used for data exploration, feature engineering, model training, and exporting the trained model.

### **Steps:**
1. **Dataset Collection**:
   - Dataset: [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data)
   - The dataset contains emotional speech files labeled with unique identifiers.

2. **Data Labeling**:
   - Labeling based on the filename structure provided in the dataset.

3. **Feature Extraction**:
   - **Techniques Used**:
     - **MFCC** (Mel-frequency cepstral coefficients)
     - **STFT** (Short-time Fourier Transform)
   - **Library**: [Librosa](https://librosa.org/)

4. **Feature Selection**:
   - **Technique**: Recursive Feature Elimination (RFE)
   - **Model Used**: RandomForestClassifier
   - **Library**: scikit-learn

5. **Model Selection and Training**:
   - **Algorithms Tried**: 
     - Random Forest
     - Support Vector Classifier (SVC)
     - K-Nearest Neighbors (KNN)
     - Gradient Boosting
     - Convolutional Neural Networks (CNN)
   - **Optimization**: Train-Test Split (80%-20%), Adam Optimizer
   - Final Model: **CNN**

6. **Exporting the Model**:
   - Exported as `.h5` file with architecture (`SpeechEmotionCNNModel.json`) and weights (`SpeechEmotionCNNModel.weights.h5`).

### **Stacks Used**:
- `glob`, `os`, `numpy`, `pandas`, `matplotlib`, `librosa`, `sklearn`, `keras`

---

## **2. Django Web Application**
The Django-based web application provides a simple UI for users to upload audio files and analyze them for emotional content.

### **Features**:
- **Upload Audio File**: Users can upload `.wav` audio files.
- **Analyze**: Triggers the loading of the pre-trained model and processes the uploaded file.
- **Emotion Prediction**: Displays the predicted emotion (e.g., Happy, Angry, Sad, etc.).
- **History**: Maintains History of predicted inputs.

---

## **Setup and Installation**

### **A. Setting up the Notebook**
**Windows**

```shell
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac**

```shell
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

Run the Notebook: Open SpeechEmotionDetection.ipynb in your preferred Jupyter environment and follow the steps to train and export the model.

### **B. Setting up the Django Web Application**
```shell
cd speech_emotion_recog
python manage.py migrate
python manage.py runserver
```

Access the Web App: Open your browser and navigate to http://127.0.0.1:8000.