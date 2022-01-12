import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from tensorflow.compat.v1 import ConfigProto
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import os
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import defaultdict, deque
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from PIL import Image
# from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.5, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
H = 0
W = 0
margin = 0


def del_id(pts):
    previous_time = datetime.datetime.now() - datetime.timedelta(hours=0, minutes=5)
    del_ids = []
    for track_id, value in pts.items():
        # check timestamp of the first track_id
        # delete that track_id's value if timestamp is 5 minutes before
        if value[0]['timestamp'] < previous_time:
            del_ids.append(track_id)

    for del_id in del_ids:
        del pts[del_id]
    return pts


def del_track_id(counted_ids, second_duration):
    previous_time = datetime.datetime.now() - datetime.timedelta(hours=0, minutes=0, seconds=second_duration)
    for counted_id in list(counted_ids):
        if counted_id['timestamp'] < previous_time:
            counted_ids.remove(counted_id)

    return counted_ids


# right or bottom
def checkDirection1(pts, x_c, y_c, direction):
    # check right
    if direction:
        condition = len(pts) >= 2 and pts[-1]['x_c'] > W // 2 + margin and pts[-2]['x_c'] <= W // 2 + margin
        mean_pts = np.mean([p['x_c'] for p in pts])
        move = x_c - mean_pts
    # check bottom
    else:
        condition = len(pts) >= 2 and pts[-1]['y_c'] > H // 2 + margin and pts[-2]['y_c'] <= H // 2 + margin
        mean_pts = np.mean([p['y_c'] for p in pts])
        move = y_c - mean_pts

    if condition and move > 0:
        return True
    else:
        return False


# left or top
def checkDirection2(pts, x_c, y_c, direction):
    # check left
    if direction:
        condition = len(pts) >= 2 and pts[-1]['x_c'] < W // 2 - margin and pts[-2]['x_c'] >= W // 2 - margin
        mean_pts = np.mean([p['x_c'] for p in pts])
        move = x_c - mean_pts
    # check top
    else:
        condition = len(pts) >= 2 and pts[-1]['y_c'] < H // 2 - margin and pts[-2]['y_c'] >= H // 2 - margin
        mean_pts = np.mean([p['y_c'] for p in pts])
        move = y_c - mean_pts

    if condition and move < 0:
        return True
    else:
        return False


def main(_argv):
    global H, W
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.7

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except Exception as e:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    pts = defaultdict(list)
    direction = 0
    direction_count_1 = 0
    direction_count_2 = 0
    counted_ids = deque(maxlen=10000)
    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        H, W = frame.shape[0:2]
        frame_num += 1
        # print('Frame #: ', frame_num)
        # frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        #
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']
        # allowed_classes = ['car', 'bus', 'truck']
        # allowed_classes = ['cow']

        # draw center line
        if direction:
            cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 255), 2)
        else:
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 255), 2)

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # draw score
        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2) + "%"
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            if len(classes) > 0:
                cv2.putText(frame, score, (int(bbox[0]), int(bbox[3])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id
            # calculate center of object
            x_c = int((bbox[0] + bbox[2]) / 2)
            y_c = int((bbox[1] + bbox[3]) / 2)
            cv2.circle(frame, center=(x_c, y_c), radius=2, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_4, shift=0)
            pts[track_id].append({'timestamp': datetime.datetime.now(), 'x_c': x_c, 'y_c': y_c})
            track_ids = [counted_id['track_id'] for counted_id in counted_ids]
            if track_id not in track_ids and checkDirection1(pts[track_id], x_c, y_c, direction):
                # counted_ids.append(track_id)
                counted_ids.append({'timestamp': datetime.datetime.now(), 'track_id': track_id})
                direction_count_1 += 1
                print('direction_1 was changed!')
            track_ids = [counted_id['track_id'] for counted_id in counted_ids]
            if track_id not in track_ids and checkDirection2(pts[track_id], x_c, y_c, direction):
                # counted_ids.append(track_id)
                counted_ids.append({'timestamp': datetime.datetime.now(), 'track_id': track_id})
                direction_count_2 += 1
                print('direction_2 was changed!')

            class_name = track.get_class()
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        fps = f"%.2f" % fps
        # print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # check time to remove not counted id
        pts = del_id(pts)
        counted_ids = del_track_id(counted_ids, second_duration=3)
        if direction == 1:
            info = [
                ("L->R", direction_count_1),
                ("L<-R", direction_count_2),
                # ("T->B", direction_count_1),
                # ("T<-B", direction_count_2),
                # ("out", direction_count_1),
                # ("in", direction_count_2),
                ("FPS", fps),
            ]
        else:
            info = [
                # ("L->R", direction_count_1),
                # ("L<-R", direction_count_2),
                ("T->B", direction_count_1),
                ("T<-B", direction_count_2),
                # ("out", direction_count_1),
                # ("in", direction_count_2),
                ("FPS", fps),
            ]

        # Display the monitor result
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(result, text, (10, ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
