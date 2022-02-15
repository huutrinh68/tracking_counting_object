import cv2
import math
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
from core import preprocessing
from collections import defaultdict, deque
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
# tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
# tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
# tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
#                      tracker_output_format='mot_challenge')

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

H = 0
W = 0
margin = 0


def del_id(pts):
    previous_time = datetime.datetime.now() - datetime.timedelta(hours=0, minutes=0, seconds=20)
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


def drawMovementFlow(pts, image):
    for track_id, value in pts.items():
        # get random color
        color = list(np.random.choice(range(256), size=3))
        # convert int64 to int, then get tuple
        color = tuple(list(map(int, color)))
        tracjectories = [(v['x_c'], v['y_c']) for v in value]
        tracjectories = tracjectories[::-1]

        for i in range(1, len(tracjectories)):
            cv2.line(image, tracjectories[i - 1], tracjectories[i], color, 1)

    return image


def distance(point1, point2):
    return math.sqrt((point2[1] - point1[1])**2 + (point2[0] - point1[0])**2)


def where_is_point(points, track_points):
    ok_id = []
    ng_id = []
    for track_id, value in track_points.items():
        tracjectories = [(v['x_c'], v['y_c']) for v in value]
        start_point = tracjectories[0]
        stop_point = tracjectories[-1]

        start_distance = []
        start_distance.append(distance(start_point, points['pointA']))
        start_distance.append(distance(start_point, points['pointB']))
        start_distance.append(distance(start_point, points['pointC']))
        start_distance.append(distance(start_point, points['pointD']))
        min_index = start_distance.index(min(start_distance))
        start_name = list(points.keys())[min_index]
        print("start_name: {}".format(start_name))

        stop_distance = []
        stop_distance.append(distance(stop_point, points['pointA']))
        stop_distance.append(distance(stop_point, points['pointD']))
        stop_distance.append(distance(stop_point, points['pointC']))
        stop_distance.append(distance(stop_point, points['pointD']))
        min_index = stop_distance.index(min(stop_distance))
        stop_name = list(points.keys())[min_index]
        print("stop_name: {}".format(stop_name))

        if (start_name == 'pointA' and stop_name == 'pointB') or \
                (start_name == 'pointB' and stop_name == 'pointA') or \
                (start_name == 'pointC' and stop_name == 'pointD') or \
                (start_name == 'pointD' and stop_name == 'pointC'):
            ok_id.append(track_id)
        else:
            ng_id.append(track_id)

    return ok_id, ng_id


def yxyx2xywh(yxyx):
    if len(yxyx.shape) == 2:
        w, h = yxyx[:, 3] - yxyx[:, 1] + 1, yxyx[:, 2] - yxyx[:, 0] + 1
        xywh = np.concatenate((yxyx[:, 1, None], yxyx[:, 0, None], w[:, None], h[:, None]), axis=1)
        return xywh.astype("int")
    elif len(yxyx.shape) == 1:
        (top, left, bottom, right) = yxyx
        width = right - left + 1
        height = bottom - top + 1
        return np.array([left, top, width, height]).astype('int')
    else:
        raise ValueError("Input shape not compatible.")


def main(_argv):
    global H, W
    nms_max_overlap = 0.7
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
        # out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (height, width))

    pts = defaultdict(list)
    direction = 1
    direction_count_1 = 0
    direction_count_2 = 0
    counted_ids = deque(maxlen=10000)
    frame_num = 0
    # while video is running
    stop_points = {"pointA": (28, 480), "pointB": (1032, 250), "pointC": (871, 635), "pointD": (470, 1)}
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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

        allowed_classes = ['person']

        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        # bboxes, scores, classes = NMS(bboxes, scores, classes)
        indices = preprocessing.non_max_suppression(bboxes, classes, nms_max_overlap, scores)
        bboxes = np.array([bboxes[i] for i in indices])
        scores = np.array([scores[i] for i in indices])
        classes = np.array([classes[i] for i in indices])

        # update tracker
        tracks = tracker.update(bboxes, scores, classes)

        # update tracks
        for track in tracks:
            # track_id
            track_id = track[1]
            # x coordinate
            x_coord = track[2]
            # y coordinate
            y_coord = track[3]
            # width
            width = track[4]
            # height
            height = track[5]
            # x_center
            x_c = int(x_coord + 0.5 * width)
            # y_center
            y_c = int(y_coord + 0.5 * height)

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

        for bb, conf, cid, track in zip(bboxes, scores, classes, tracks):
            track_id = track[1]
            # draw result on screen
            color = colors[int(track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.circle(frame, center=(x_c, y_c), radius=2, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_4, shift=0)
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])), color, 2)
            label = "ID:{} score:{:.3f}".format(track_id, conf)
            (label_width, label_height), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(bb[1], label_height)
            cv2.putText(frame, label, (int(bb[0]), int(y_label)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        fps = f"%.2f" % fps
        # print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.circle(result, stop_points['pointA'], 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_8, shift=0)
        cv2.circle(result, stop_points['pointB'], 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_8, shift=0)
        cv2.circle(result, stop_points['pointC'], 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_8, shift=0)
        cv2.circle(result, stop_points['pointD'], 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_8, shift=0)

        # check time to remove not counted id
        pts = del_id(pts)

        # counted_ids = del_track_id(counted_ids, second_duration=3)
        ok_id, ng_id = where_is_point(stop_points, pts)
        if direction == 1:
            info = [
                ("ok_id", ok_id),
                ("ng_id", ng_id),
                ("FPS", fps),
            ]
        else:
            info = [
                ("ok_id", ok_id),
                ("ng_id", ng_id),
                ("FPS", fps),
            ]

        result = drawMovementFlow(pts, result)
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
