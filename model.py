from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import json
"""
Safety vest, No hardhat, gloves, Mask, vehicle
"""
image_path = "resized_test_image_2.jpg"
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="AonDlnwLMihldiyHw3P9"
)

results = CLIENT.infer(image_path, model_id="construction-site-safety/27")
print(json.dumps(results, indent=4))
#Draw the results on the image 
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for result in results["predictions"] :
    x, y, width, height = result['x'], result['y'], result['width'], result['height']
    x1 = x - width / 2
    y1 = y - height / 2
    x2 = x1 + width
    y2 = y1 + height
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
image.show()
image.save(f"resized_test_image_2_inference.jpg")

