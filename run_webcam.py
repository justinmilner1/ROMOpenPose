import argparse
import logging
import time
from datetime import datetime
import threading

import cv2
import numpy as np
import math
from math import atan2, pi
import statistics
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import csv

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def angle(A, B, C, /):
    Ax, Ay = A[0]-B[0], A[1]-B[1]
    Cx, Cy = C[0]-B[0], C[1]-B[1]
    a = atan2(Ay, Ax)
    c = atan2(Cy, Cx)
    if a < 0: a += pi*2
    if c < 0: c += pi*2
    return (pi*2 + c - a) if a > c else (c - a)

def get_joint_angle(human, bp1, bpcenter, bp2, input):
    center_x = human.body_parts[bpcenter].x*image.shape[1]
    center_y = human.body_parts[bpcenter].y * image.shape[0]
    bp1_x = human.body_parts[bp1].x * image.shape[1]
    bp1_y = human.body_parts[bp1].y * image.shape[0]
    bp2_x = human.body_parts[bp2].x * image.shape[1]
    bp2_y = human.body_parts[bp2].y * image.shape[0]
    angle = getAngle([bp1_x, bp1_y], [center_x, center_y], [bp2_x, bp2_y])
    if angle > 185:
        return 360 - angle
    else:
        return angle

    # if input == 1 or input == 2:
    #     if angle > 270:
    #         return 360 - angle
    #     else:
    #         return angle
    # if input == 3 or input ==4:
    #     if angle > 200:
    #         return


def get_joint_points(joint):
    bp1 = None
    bpcenter = None
    bp2 = None
    joint_name = None

    joint = int(joint)
    if joint == 1 or joint == 2:
        bp1 = 10
        bpcenter = 8
        bp2 = 13
        joint_name = 'Split'
    elif joint == 3:
        bp1 = 5
        bpcenter = 6
        bp2 = 7
        joint_name = 'Elbow'
    elif joint == 4:
        bp1 = 2
        bpcenter = 3
        bp2 = 4
        joint_name = 'Elbow'
    elif joint == 5:
        bp1 = 12
        bpcenter = 13
        bp2 = 14
        joint_name = 'Knee'
    elif joint == 6:
        bp1 = 9
        bpcenter = 10
        bp2 = 11
        joint_name = 'Knee'
    return bp1, bpcenter, bp2, joint_name

def get_input():
    global end_flag
    end_flag = True
    end_flag = input("Press a key to end loop")
    # thread doesn't continue until key is pressed
    end_flag=False
    print('flag is now:', end_flag)



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_v2_small', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    #taking user input
    input = input("Enter 1 for side splits: " + '\n'+
                  "      2 for front splits: " + '\n'+
                  "      3 for left elbow extension: " + '\n'+
                  "      4 for right elbow extension: " + '\n' +
                  "      5 for left knee extension: " + '\n' +
                  "      6 for right knee extension: " )
    bp1, bpcenter, bp2, joint_name = get_joint_points(input)

    angle_list = []
    frame_number = 0
    display_angle = 0
    try:
        while True:
            frame_number += 1
            ret_val, image = cam.read()

            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            try:
                for human in humans:
                    angle_list.append(get_joint_angle(human, bp1, bpcenter, bp2, input))
                    display_angle = statistics.mean(angle_list[-5:])
                    if len(angle_list) > 1000 == 0:
                        print("deleted")
                        del angle_list[:500]
            except Exception as err:
                pass
                #print("Error: ", err)
            finally:
                cv2.putText(image, "%s angle: %f" % (joint_name, display_angle),
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            # cv2.putText(image,
            #             "Number of humans: %f" % len(humans), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)


            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            logger.debug('finished+')
            if frame_number == 1:
                print("Press cntl C to end")
    finally:
        #finding best angle achieved
        slide_nums = 5
        best_angle = None
        if input == 1 or input == 2: #maximize
            best_angle = 0
            for index in range(5, len(angle_list)):
                if statistics.mean(angle_list[index-slide_nums:index+slide_nums]) > best_angle:
                    best_angle = statistics.mean(angle_list[index-slide_nums:index+slide_nums])
        else:
            best_angle = 180
            for index in range(5, len(angle_list)):
                print("list: ", angle_list)
                print("index: ", index, " | slide num: ", slide_nums)
                print("Len: ", len(angle_list), "| mean: ", statistics.mean(angle_list[index - slide_nums:index + slide_nums]))
                if statistics.mean(angle_list[index - slide_nums:index + slide_nums]) < best_angle:
                    best_angle = statistics.mean(angle_list[index - slide_nums:index + slide_nums])
        print("Best angle: ",  best_angle)

        #writing results to csv
        row = str(datetime.today().strftime('%Y-%m-%d')) + ', ' + str(joint_name) + ', ' + str(best_angle) + '\n'
        print("row: ", row)
        with open('./ROM_records.csv', 'a') as out:
            out.write(row)
            out.close()
            print("row written")

        cv2.destroyAllWindows()
