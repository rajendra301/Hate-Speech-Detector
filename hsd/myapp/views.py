import os
import json
import joblib
import re
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# --- SETUP: LOAD MODELS ONCE ---
# Calculate the path to the 'hsd' folder (where manage.py is)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths to the model files
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model_nb.pkl') # Change to 'model_lr.pkl' if using Logistic Regression

print(f"DEBUG: Looking for models at: {BASE_DIR}")

# Global variables to hold the brain
vectorizer = None
model = None

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    print("SUCCESS: ML Models loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. {e}")
    # We don't exit here so the server can still run, 
    # but predictions will fail with a clear error.

# --- VIEWS ---

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def predict_hate_speech(request):
    global vectorizer, model # explicitly refer to the globals
    
    if request.method == 'POST':
        try:
            # 1. Check if models exist
            if vectorizer is None or model is None:
                print("DEBUG: Prediction failed because models are not loaded.")
                return JsonResponse({'error': 'Models missing on server. Check terminal logs.'}, status=500)

            # 2. Get data
            data = json.loads(request.body)
            user_text = data.get('text', '')
            print(f"DEBUG: Processing text: '{user_text}'")

            if not user_text:
                return JsonResponse({'error': 'No text provided'}, status=400)

            # 3. Clean Data (Must match training!)
            clean_text = user_text.lower()
            clean_text = re.sub(r'[^\w\s]', '', clean_text)

            # 4. Predict
            text_vec = vectorizer.transform([clean_text])
            prediction = model.predict(text_vec)[0]
            probability = model.predict_proba(text_vec)[0][1]

            print(f"DEBUG: Result -> Hate: {prediction}, Confidence: {probability:.2f}")

            return JsonResponse({
                'isHateSpeech': bool(prediction == 1),
                'confidence': float(probability)
            })

        except Exception as e:
            print(f"DEBUG ERROR: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)