import argparse

import cv2
import torch
from utils.datasets import LoadImages

from utils.general import (
    non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.general import *
from flask import Flask, escape, request, jsonify, make_response, render_template
import json
import os
import time
import threading
from utils.plots import plot_one_box

mutex = threading.Lock()
app = Flask(__name__)
app.config["DEBUG"] = True
app.config['SECRET_KEY'] = 'secret!'

imagepath = "/dev/shm/"
# imagepath = "./test/"

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    # filter removes empty strings (such as last line)
    return list(filter(None, names))


def init_detect():
    global device, model, half, opt, imgsz, names,opt
    weights, imgsz, cfg, names = opt.weights, opt.img_size, opt.cfg, opt.names
    names = load_classes(names)
    # Initialize
    device = select_device(opt.device)

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None


@app.route("/detect", methods=['POST', "GET"])
def aidetect():
    
    fileName = str(time.time())
    img = request.files.get('file')

    if not os.path.isdir(imagepath+"photo/"):
        os.mkdir(imagepath+"photo/")

    filename = imagepath+"photo/"+fileName+".jpg"
    img.save(filename)
    mutex.acquire()  # 鎖定
    try:
        data = detect(filename)
    except:
        print("aidetect error")
    mutex.release()
    return jsonify(data)


def detect(filename,saveoutput=True):
    global device, model, half, opt, imgsz, names
    dataset = LoadImages(filename, img_size=imgsz, auto_size=64)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=False)
        t2 = time_synchronized()

        # # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)
        plist = list()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            p, s, im0 = path, '', im0s
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                if saveoutput:
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                for *xyxy, conf, cls in det:
                    if names[int(cls)] != "person":
                        continue
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh
                    print(names[int(cls)]+":"+str(xywh))
                    plist.append({"name": names[int(cls)], "xywh": xywh})
                    if saveoutput:
                        # 繪製
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=[111,111,111], line_thickness=1)
        if saveoutput:                        
            cv2.imwrite("tmp.jpg", im0)

        # 移除 暫存檔案
        try:
            os.remove(filename)
        except OSError as e:
            print(e)
        else:
            print("File is deleted successfully")
        return plist


def initdata():
    global opt
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--cfg', type=str,
                        default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str,
                        default='data/coco.names', help='*.cfg path')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')

    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                init_detect()
                strip_optimizer(opt.weights)
        else:
            init_detect()
    app.run(host="0.0.0.0", port=6858,debug=False, use_reloader=False)
initdata()
    
