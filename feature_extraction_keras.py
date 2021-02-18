from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model, Model
from keras.layers import Input, Lambda
from keras import backend as K
from yolo3.utils import letterbox_image
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_eval
from PIL import Image, ImageFont, ImageDraw
from keras.utils import plot_model
import numpy as np
import cv2
from yolo_keras_v2 import load_image_pixels, decode_netout, correct_yolo_boxes, do_nms, get_boxes

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0: 32, 1: 16, 2: 8}[l], w//{0: 32, 1: 16, 2: 8}[l],
                           num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


# with a Sequential model
classes_path = 'model_data/coco_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
input_shape = (416, 416)  # multiple of 32, hw
# model = create_model(input_shape, anchors, num_classes,
#  freeze_body = 2, weights_path = 'model_data/yolov3_2.h5')
model = load_model('model_data/yolov3_2.h5')
model.summary()

with open('/home/sysadmin/darknet/LSTM_Train_Images/lstm_bottle.txt') as f:
    lines = f.readlines()
    for line in lines:
        # img_name = "./" + str(line)#"/home/sysadmin/darknet/"+str(line)
        file_name = str(line.split('/')[1].strip())
        print(file_name)
        # Read image and preprocess its size
        image_path = "/home/sysadmin/darknet/LSTM_Train_Images/lstm_train_images/"+file_name
        img_name = file_name.split('.')[0]
        image = Image.open(image_path)
        boxed_image = letterbox_image(
            image, tuple(reversed(input_shape)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        # create model using anchor, class list and weight path

        index_list = [240, 241, 242]
        detection_array = np.array([])
        for index in index_list:
            get_layer_output = K.function([model.layers[0].input],
                                          [model.layers[index].output])
            layer_output = get_layer_output([image_data])[0]
            layer_output = layers.GlobalAveragePooling2D()(layer_output)
            detection = np.ctypeslib.as_array(layer_output)

            if detection_array == []:
                detection_array = detection
            else:
                detection_array = np.concatenate(
                    (detection_array, detection), axis=None)

        image, image_w, image_h = load_image_pixels(
            image_path, input_shape)
        # make prediction
        yhat = model.predict(image)
        # define the anchors
        anchors = [[116, 90, 156, 198, 373, 326], [
            30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        # define the probability threshold for detected objects
        class_threshold = 0.6
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], anchors[i],
                                   class_threshold, input_shape[0], input_shape[1])
        # correct the sizes of the bounding boxes for the shape of the image
        correct_yolo_boxes(boxes, image_h, image_w,
                           input_shape[0], input_shape[1])
        # suppress non-maximal boxes
        do_nms(boxes, 0.5)
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
        for i in range(len(v_boxes)):
            if v_labels[i] == 'bottle':
                box = v_boxes[i]
                y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
                score = v_scores[i]/100
                v_labels = labels.index(str(v_labels[i]))
                width, height = (x2 - x1), (y2 - y1)
                cx, cy = x1+width/2, y1+height/2
                output_array = np.array(
                    [cx/640, cy/480, width/640, height/480, score])
                print(output_array)
            else:
                output_array = np.array([None, None, None, None, None])
        detection_array = np.concatenate(
            (detection_array, output_array), axis=None)
        print(detection_array.shape)
        # break
        np.save("/home/sysadmin/darknet/LSTM_Train_Images/LSTM_npy/" +
                img_name+'.npy', detection_array)
        print('npy saved')
