from flask import Flask, request, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Path to your TFLite model
MODEL_PATH = "/home/ec2-user/lyricGen/cartoon-flask-app/model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

def preprocess_image(image, input_shape):
    image = image.convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image).astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(image, axis=0)

def postprocess_image(output_data):
    output_data = (output_data + 1.0) * 127.5
    output_data = np.clip(output_data, 0, 255).astype(np.uint8)
    return Image.fromarray(output_data[0])

def convert_image(input_image, output_image_path):
    input_data = preprocess_image(input_image, input_shape)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_image = postprocess_image(output_data)
    
    print(f"Saving output image to {output_image_path}")
    output_image.save(output_image_path)

@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Create directory if it doesn't exist
        os.makedirs('static', exist_ok=True)
        # Input and output paths
        input_path = os.path.join('static', 'input.jpg')
        output_path = os.path.join('static', 'output.jpg')
        
        try:
            print(f"Saving input file to {input_path}")
            file.save(input_path)
            
            print(f"Opening input file from {input_path}")
            input_image = Image.open(input_path)
            
            print(f"Converting image and saving to {output_path}")
            convert_image(input_image, output_path)
            
            if os.path.exists(output_path):
                print(f"Output file exists: {output_path}")
            else:
                print(f"Output file does not exist: {output_path}")
            
            print(f"Sending output file from {output_path}")
            return send_file(output_path, mimetype='image/jpeg')
        except Exception as e:
            print(f"Error: {e}")
            return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
