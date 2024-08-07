from flask import Flask, request, render_template, send_file, abort
import pandas as pd
import os
from werkzeug.utils import secure_filename
import zipfile
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'zip'}
app.config['IMAGE_FOLDER'] = os.path.normpath('C:/Users/user/Desktop/mywork/Prediscan/uploads/extracted_images/Retinal Image Folder')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        excel_file = request.files.get('excel')
        zip_file = request.files.get('zip')
        
        if not excel_file or not zip_file:
            abort(400, "Both Excel and ZIP files are required.")
        
        if not (allowed_file(excel_file.filename, {'xlsx'}) and allowed_file(zip_file.filename, {'zip'})):
            abort(400, "Invalid file type. Please upload an Excel file and a ZIP file.")
        
        excel_filename = secure_filename(excel_file.filename)
        zip_filename = secure_filename(zip_file.filename)
        
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
        
        excel_file.save(excel_path)
        zip_file.save(zip_path)
        
        return process_files(excel_path, zip_path)
    
    return render_template('index.html')

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Contents of {extract_to}:")
    for root, dirs, files in os.walk(extract_to):
        for name in files:
            print(os.path.join(root, name))

model = load_model(os.path.normpath('C:/Users/user/Desktop/mywork/Prediscan/dr_model.h5'))
print(model.summary()) 


def process_image(image_path):
    try:
        with Image.open(image_path) as image:
            image = image.resize((224, 224))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)
            
         
            return prediction.flatten()
    except Exception as e:
        print(f"Error processing  image {image_path}: {e}")
        return None

def update_image_paths(df):
    image_folder = app.config['IMAGE_FOLDER']
    
    for index, row in df.iterrows():
        left_image_filename = row.get('Left Image ID', '')
        right_image_filename = row.get('Right Image ID', '')

        left_image_path = os.path.normpath(os.path.join(image_folder, f"{left_image_filename}.png"))
        right_image_path = os.path.normpath(os.path.join(image_folder, f"{right_image_filename}.png"))

        print(f"Generated left image path: {left_image_path}")  
        print(f"Generated right image path: {right_image_path}")  

        if not os.path.isfile(left_image_path):
            print(f"Warning: {left_image_path} does not exist")
            left_image_path = ''
        if not os.path.isfile(right_image_path):
            print(f"Warning: {right_image_path} does not exist")
            right_image_path = ''

        df.at[index, 'Left Image Path'] = left_image_path
        df.at[index, 'Right Image Path'] = right_image_path
    
    return df

def update_excel(excel_path):
    df = pd.read_excel(excel_path)
    df = update_image_paths(df)
    print("Columns in Excel file:", df.columns) 

    left_eye_columns = [f'DR Predicted (Left) {i}' for i in range(5)]
    right_eye_columns = [f'DR Predicted (Right) {i}' for i in range(5)]

    for index, row in df.iterrows():
        left_eye_image_path = row.get('Left Image Path', '')
        right_eye_image_path = row.get('Right Image Path', '')
        if pd.isna(left_eye_image_path) or not isinstance(left_eye_image_path, str) or not os.path.isfile(left_eye_image_path):
            print(f"Missing or invalid left eye image path for row {index}")
            continue
        if pd.isna(right_eye_image_path) or not isinstance(right_eye_image_path, str) or not os.path.isfile(right_eye_image_path):
            print(f"Missing or invalid right eye image path for row {index}")
            continue

        print(f"Processing left eye image: {left_eye_image_path}")  # Debugging line
        left_eye_prediction = process_image(left_eye_image_path)
        
        print(f"Processing right eye image: {right_eye_image_path}")  # Debugging line
        right_eye_prediction = process_image(right_eye_image_path)

        if left_eye_prediction is not None:
            for i, prob in enumerate(left_eye_prediction):
                df.at[index, left_eye_columns[i]] = prob
        if right_eye_prediction is not None:
            for i, prob in enumerate(right_eye_prediction):
                df.at[index, right_eye_columns[i]] = prob

    updated_excel_path = excel_path.replace('.xlsx', '_updated.xlsx')
    df.to_excel(updated_excel_path, index=False)
    return updated_excel_path


def process_files(excel_path, zip_path):
    extract_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_images')
    os.makedirs(extract_folder, exist_ok=True)
    extract_zip(zip_path, extract_folder)
    updated_excel_path = update_excel(excel_path)
    return send_file(updated_excel_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)