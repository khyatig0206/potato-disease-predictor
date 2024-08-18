from PIL import Image
import numpy as np
import json
import boto3

# Initialize the boto3 client for SageMaker Runtime
client = boto3.client('sagemaker-runtime')
CLASS_NAMES = ['Early Blight', 'Healthy', 'Late Blight']

# Your SageMaker endpoint name
endpoint_name = 'tensorflow-inference-2024-08-18-09-24-14-553'

# Load and preprocess the image
image_path = 'D:/potato-disease/training/Late_Blight/Late_Blight_8.jpg'  # Replace with the path to your image
image = Image.open(image_path)
image = image.resize((256, 256))  # Resize image to 256x256

# Convert the image to a numpy array
image_array = np.array(image)

# Normalize 
image_array = image_array / 255.0

# Expand dimensions to match the model's expected input shape
image_batch = np.expand_dims(image_array, axis=0)

payload = image_batch.tolist()

# Convert the payload to JSON format
payload_json = json.dumps(payload)

# Send the request to the SageMaker endpoint
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=payload_json
)

# Parse the response
response_body = response['Body'].read().decode('utf-8')  # Extract the body and decode it
# prediction = json.loads(response_body)['predictions']
# predicted_class = CLASS_NAMES[np.argmax(prediction)]
# confidence = round(100 * np.max(prediction), 2)
# print(predicted_class, confidence)
print(response_body)