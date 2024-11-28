from flask import Flask, render_template, request, jsonify
import pickle, os
from tensorflow.keras.preprocessing import image
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

upload_folder = 'uploads'
allowed_extensions = {'png','jpeg','jpg'}
app.config['upload_folder'] = upload_folder


model = pickle.load(open('Enet_model.pkl','rb'))

def is_File_Allowed(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in allowed_extensions


@app.route('/',methods=['GET'])

def main():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])

def submit():
    if 'file' not in request.files:
        return jsonify({'error': 'Image is not uploaded with the key as file'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file is selected or uploaded by the user'})
    
    if file and is_File_Allowed(file.filename):

        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config['upload_folder'],filename)

        file.save(file_path)

        try:
            img = Image.open(file_path)

        except IOError:
            print("This file is not a valid image and could not load that image")

        img = image.load_img(file_path, target_size=(224, 224)) 

        img_array = image.img_to_array(img)  # Convert image to numpy array

        
        img_array = np.expand_dims(img_array, axis=0)

        img_array = img_array / 255.0  # Normalize if model expects values in [0, 1]

        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction, axis=1)[0]

        # return jsonify({'Image Belong to the class =' : str(predicted_class)}), 200
        # if predicted_class == 0:
        #     return "Not a Diabetic retinopathy retinopathy Image with a Probability of", 200
        # else:
        #     return "Diabetic Retinopathy", 200
        if prediction[0][0] > prediction[0][1]:
            return "Not having a Diabetic retinopathy with a probability of {:.3f} %".format(prediction[0][0]*100), 200
        else:
            return "Diabetic Retinopathy with having a probability of {:.3f} %".format(prediction[0][1]*100), 200
    
    else:
        return jsonify({'error': 'Invalid File Format'}), 400


if __name__ == "__main__":
    app.run(port=3000, debug=True)