from flask import Flask, request, render_template, url_for
import os
import gc
import cv2
from ultralytics import YOLO
import random

# Initialize Flask app
app = Flask(__name__)

# Path to your YOLO weight file (update the path as per your setup)
model = YOLO('runs/train/teeth_det/weights/best.pt')

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'

# Ensure the required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Clear memory and garbage collection
gc.collect()

# Custom order for teeth numbering based on the given X-ray image order
custom_teeth_order_UTN = ['T1', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16',
                           'T17', 'T18', 'T19', 'T2', 'T20', 'T21', 'T22', 'T23',
                           'T24', 'T25', 'T26', 'T27', 'T28', 'T29', 'T3', 'T30',
                           'T31', 'T32', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']

custom_teeth_order_FDI = ['11', '32', '33', '34', '35', '36', '37', '38',  # Maxillary Right
                           '39', '41', '42', '21', '43', '44', '45', '46',  # Maxillary Left
                           '47', '14', '15', '16', '17', '18', '22', '19',  # Mandibular Left
                           '12', '13', '23', '24', '25', '26', '27', '31']

# Mapping class IDs from YOLO (0 to 31) to custom teeth order
teeth_numbers_UTN = {i: custom_teeth_order_UTN[i] for i in range(32)}
teeth_numbers_FDI = {i: custom_teeth_order_FDI[i] for i in range(32)}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if files are uploaded
        if 'files[]' not in request.files:
            return "No file part"

        files = request.files.getlist('files[]')
        if len(files) == 0 or len(files) > 10:
            return "Please upload between 1 and 10 images."

        result_paths = []
        teeth_format = request.form.get('teeth_format')  # Get the selected teeth format

        for file in files:
            # Save the uploaded file
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Load the image using OpenCV
            img = cv2.imread(file_path)

            # YOLO detection
            results = model.predict(source=file_path,
                                    imgsz=(640, 640),
                                    conf=0.35,
                                    iou=0.45,
                                    max_det=32,  # Limit to 32 detections for teeth
                                    device='',
                                    save_txt=False,
                                    save_conf=False,
                                    save=False,
                                    save_crop=False,
                                    show=False,
                                    line_width=2)
            print (results)
            # Assign random colors for each detection
            colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(32)]

            # Get bounding boxes and draw on the image
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = box.conf[0].item()  # Confidence score
                    cls = int(box.cls[0].item())  # Class ID (for teeth numbering)

                    # Assign teeth number based on the selected format
                    if teeth_format == 'FDI':
                        teeth_number = teeth_numbers_FDI.get(cls, f'T{cls + 1}')
                    else:
                        teeth_number = teeth_numbers_UTN.get(cls, f'T{cls + 1}')

                    # Choose a random color for each box
                    color = colors[cls % 32]

                    # Draw bounding box and label with teeth number and confidence score
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f'{teeth_number} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Save the resulting image
            result_path = os.path.join(RESULT_FOLDER, 'detected_' + file.filename)
            cv2.imwrite(result_path, img)
            result_paths.append('results/detected_' + file.filename)

        # Return the processed images for display
        return render_template('index.html', result_paths=result_paths)

    return render_template('index.html')

@app.route('/view-image/<path:image_path>')
def view_image(image_path):
    """Route to view the larger image."""
    return render_template('view_image.html', result_paths=[image_path])

if __name__ == '__main__':
    app.run(debug=True)