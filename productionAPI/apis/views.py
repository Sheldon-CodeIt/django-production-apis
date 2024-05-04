from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import json
import os
import pickle
import numpy as np
import re

ml_model = pickle.load(open("./ml-models/mental_health_prediction/mental_health_prediction.sav",'rb'))
        
with open('./ml-models/mental_health_prediction/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def Home(request):
    return HttpResponse("Home")


@csrf_exempt
def PredictMentalHealth(request):
    if request.method == 'POST':

        try:
            # Parse request data
            data = json.loads(request.body)
            input_features = data.get('input_features')  # Using get() method to handle missing key gracefully
            if input_features is None:
                raise KeyError('Invalid input format. "input_features" key missing.')
            
            input_array = np.array(input_features)
            input_reshaped = input_array.reshape(1, -1)
            scaled_input = scaler.transform(input_reshaped)
            prediction = ml_model.predict(scaled_input)


            response_data = {
                'prediction': int(prediction[0])
            }

            return JsonResponse(response_data)

        except KeyError:
            return JsonResponse({'error': 'Invalid input format. "input_features" key missing.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    else:
        return JsonResponse({'error': 'Only POST requests are supported.'}, status=400)

