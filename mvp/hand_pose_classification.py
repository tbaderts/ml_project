"""
this application is largely based on code provided by Nvidia and is used for educational purposes only
https://github.com/NVIDIA-AI-IOT/trt_pose
https://github.com/NVIDIA-AI-IOT/trt_pose_hand
"""

from utils.preprocessdata import preprocessdata
import torch
import torchvision.transforms as transforms
from torch2trt import TRTModule
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import PIL.Image
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import trt_pose.coco
import math
import os
import numpy as np
import traitlets
import pickle
import sys
import argparse

sys.path.append("../")
svm_filename = 'model/svmmodel.sav'
gesture_definition = 'preprocess/gesture.json'

# optionally use re-trained classifier & classes for demo
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--new", help="use new classifier & classes", action='store_true')
args = parser.parse_args()
if args.new:
    print("using new classifier & classes")
    svm_filename = 'model/svmmodel_new.sav'
    gesture_definition = 'preprocess/gesture_new.json'


WIDTH = 224
HEIGHT = 224
MODEL_WEIGHTS = 'model/hand_pose_resnet18_att_244_244.pt'
OPTIMIZED_MODEL = 'model/hand_pose_resnet18_att_244_244.trt'
device = torch.device('cuda')


with open('preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)

num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])
topology = trt_pose.coco.coco_category_to_topology(hand_pose)
model = trt_pose.models.resnet18_baseline_att(
    num_parts, 2 * num_links).cuda().eval()
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

if not os.path.exists(OPTIMIZED_MODEL):
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    import torch2trt
    model_trt = torch2trt.torch2trt(
        model, [data], fp16_mode=True, max_workspace_size=1 << 25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


parse_objects = ParseObjects(
    topology, cmap_threshold=0.12, link_threshold=0.15)
draw_objects = DrawObjects(topology)


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


preprocessdata = preprocessdata(topology, num_parts)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf'))
clf = pickle.load(open(svm_filename, 'rb'))


with open(gesture_definition, 'r') as f:
    gesture = json.load(f)
gesture_type = gesture["classes"]


def draw_joints(image, joints):
    count = 0
    color = (255, 255, 255)
    for i in joints:
        if i == [0, 0]:
            count += 1
    if count >= 3:
        return
    for i in joints:
        cv2.circle(image, (i[0], i[1]), 2, color, 2)
    cv2.circle(image, (joints[0][0], joints[0][1]), 2, color, 2)
    for i in hand_pose['skeleton']:
        if joints[i[0]-1][0] == 0 or joints[i[1]-1][0] == 0:
            break
        cv2.line(image, (joints[i[0]-1][0], joints[i[0]-1][1]),
                 (joints[i[1]-1][0], joints[i[1]-1][1]), color, 2)


def scale_joints(joints, w_scale, h_scale):
    for i in joints:
        i[0] = int(i[0] * w_scale)
        i[1] = int(i[1] * h_scale)


def execute(frame):
    window_name = 'hand gesture classification demo'
    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
    w_scale = int(frame.shape[1])/WIDTH
    h_scale = int(frame.shape[0])/HEIGHT
    data = preprocess(resized_frame)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    joints = preprocessdata.joints_inference(
        resized_frame, counts, objects, peaks)
    dist_bn_joints = preprocessdata.find_distance(joints)
    gesture = clf.predict([dist_bn_joints, [0]*num_parts*num_parts])
    gesture_joints = gesture[0]
    scale_joints(joints, w_scale, h_scale)
    draw_joints(frame, joints)
    preprocessdata.prev_queue.append(gesture_joints)
    preprocessdata.prev_queue.pop(0)
    flipped_frame = cv2.flip(frame, 1)
    preprocessdata.print_label(
        flipped_frame, preprocessdata.prev_queue, gesture_type)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(window_name, flipped_frame)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    execute(frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
