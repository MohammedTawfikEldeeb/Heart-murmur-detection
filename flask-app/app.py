import numpy as np

from flask import Flask, render_template, request, send_file
from keras.models import load_model
import librosa
import joblib




app = Flask(__name__)

# Load the Keras model and scaler (once during application initialization)
model = load_model('model.keras')
scaler = joblib.load('scaler.h5')


def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features




#for recorded file


@app.route('/', methods=['GET'])
def home():
    return render_template('index1.html')
@app.route('/secpage', methods=['GET'])
def secpage():
    return render_template('index2.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'fileUpload' in request.files:
        try:
            # Get the uploaded file
            uploaded_file = request.files['fileUpload']

            # Save the uploaded file to a temporary location
            uploaded_file.save("temp.wav")

            # Extract features from the audio content
            prediction_features = features_extractor("temp.wav")
            prediction_features = prediction_features.reshape(1, -1)
            prediction_features = scaler.transform(prediction_features)

            # Make predictions
            predicted_probabilities = model.predict(prediction_features)
            predicted_class = np.argmax(predicted_probabilities, axis=1)
            predicted_value = predicted_class[0]

            # Adjust predicted class
            if predicted_value == 3239:
                predicted_value = -1
            else:
                predicted_value = 1

            return render_template('index2.html', predicted_value=predicted_value)
        except Exception as e:
            # Handle unexpected errors during processing
            return f"An error occurred: {str(e)}", 500
    
    else:
        return "No audio file provided", 400



if __name__ == "__main__":
    app.run(debug=True)


