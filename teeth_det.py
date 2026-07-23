import matplotlib.pyplot as plt
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from natsort import natsorted
import gc
from ultralytics import YOLO
import numpy as np
import cv2
# Clear figure and run garbage collection
gc.collect()
plt.clf()



if __name__ == '__main__':
    root = sys.path
    print("********************************* teeth_detection_main.py ***********************************")

    mk_dir = 'result'
    det_img='result/det_img/'
    ImageFolder = 'data_det/test/images'
    weight = 'runs/train/teeth_det/weights/best.pt'


    # Create necessary directories
    if not os.path.exists(mk_dir):
        os.makedirs(mk_dir)
    for sub_dir in ['det_img']:
        os.makedirs(os.path.join(mk_dir, sub_dir), exist_ok=True)


    # Process each image
    JPGFileList = natsorted(os.listdir(ImageFolder))
    print('No of test images:',len(JPGFileList))
    serial = 0

    for image_file in JPGFileList:
        print('serial:',serial+1)
        img_fname = os.path.join(ImageFolder, image_file)
        print('img_fname:',img_fname)
        TargetName = os.path.basename(img_fname)
        text = os.path.splitext(TargetName)[0]
        print('text:',text)
        img = cv2.imread(img_fname)
        img_width, img_height = img.shape[1], img.shape[0]


        # Classification with YOLO
        model = YOLO(weight)
        results = model.predict(source=img_fname,
                                imgsz=(640, 640),
                                conf=0.35,
                                iou=0.45,
                                max_det=300,
                                device='',
                                save_txt=True,
                                save_conf=True,
                                save=True,
                                save_crop=True,

                                show=True,

                                line_width=2,

                                project=os.path.join(mk_dir, 'teeth_det'),
                                name='image',
                                exist_ok=False
                                )

        print(results[0].boxes.xyxy)
        print(results[0].boxes.conf)
        print(results[0].boxes.cls)

        boxes = results[0]  # Boxes object for bounding box outputs


        serial += 1

