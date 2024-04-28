from pathlib import Path
import torch
import cv2
import numpy as np
import argparse
import pandas as pd

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from ultralyticsmod.utils.plotting import AnnotatorMeasurement

import sys
sys.path.append("C:\\Others\\yolov5")

from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.general import (
   check_img_size,
   Profile,
   increment_path,
   non_max_suppression,
   scale_boxes,
)
from utils.dataloaders import LoadImages
from ultralytics.utils.plotting import Annotator, colors, save_one_box

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 object detection script')
    
    # Device settings
    parser.add_argument('--device', type=str, default='', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    
    # Model settings
    parser.add_argument('--weights', type=str, default='yolov5/runs/train/exp8/weights/best.pt', help='model.pt path')
    
    # Data settings
    parser.add_argument('--data', type=str, default='yolov5/data/v2.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    
    # Batch size
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    
    # Source settings
    parser.add_argument('--source', type=str, default='C:\\Others\\datasets\\augments\\test\\WIN_20240404_15_41_03_Pro0000.jpg', help='file or dir')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    
    # Output settings
    parser.add_argument('--project', default='detects/', help='save results to project/name')
    parser.add_argument('--name', default='exp/', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results to *.csv')
    
    # Detection settings
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    
    # Visualization settings
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parse_args()

    # Load model
    device = select_device(opt.device)
    model = DetectMultiBackend(
        opt.weights, 
        device=device, 
        dnn=opt.dnn, 
        data=opt.data, 
        fp16=opt.half
    )
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(
        opt.source, 
        img_size=imgsz, 
        stride=stride, 
        auto=pt, 
        vid_stride=opt.vid_stride
    )
    vid_path, vid_writer = [None] * opt.bs, [None] * opt.bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else opt.bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / "labels" if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    if opt.save_csv:
        dict_det = {
            "path": [],
            "x_coor": [],
            "y_coor": [],
        }

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=opt.augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=opt.augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=opt.augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, 
                opt.conf_thres, 
                opt.iou_thres, 
                opt.classes, 
                opt.agnostic_nms, 
                max_det=opt.max_det
            )

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = AnnotatorMeasurement(im0, line_width=opt.line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                measurement = {
                    "center": {},
                    "units": {},
                    "xyxy": {},
                }
                obj_color = {}
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if opt.hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    obj_color[label] = colors(c, True)
                    
                    if label == "ref":
                        # get units of measurement
                        # measure length from center point
                        unit_per_x = np.abs(xyxy[2] - xyxy[0])/10 # width box divided by 10
                        unit_per_y = np.abs(xyxy[3] - xyxy[1])/10 # height box divided by 10
                        measurement["units"]["x"] = unit_per_x
                        measurement["units"]["y"] = unit_per_y

                    # Add bbox to image
                    c = int(cls)  # integer class
                    label_conf = None if opt.hide_labels else (names[c] if opt.hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label_conf, color=colors(c, True))
                    cx, cy = annotator.get_center(xyxy) # get center of box

                    if label not in measurement["center"]:
                        measurement["center"][label] = []
                    measurement["center"][label].append((cx, cy)) # add center to measurement

                    if label not in measurement["xyxy"]:
                        measurement["xyxy"][label] = []
                    measurement["xyxy"][label].append((xyxy[0], xyxy[1], xyxy[2], xyxy[3])) # add xyxy to measurement
                    
                im0 = annotator.result()
                center_ref = measurement["center"]["ref"] # center reference (x, y)
                unit_x, unit_y = measurement["units"]["x"], measurement["units"]["y"]

                # Write real measurement result based on unit
                for label, centers in measurement["center"].items():
                    # Only if object(s) is/are detected (also show the origin coordinate)
                    if label == "object" or label == "ref":
                        for c in centers:
                            # c: center of bounding box
                            # Write center points to image
                            im0 = cv2.circle(
                                im0, 
                                (int(c[0]),int(c[1])), 
                                radius=0, 
                                color=obj_color[label], 
                                thickness=5
                            )
                            # estimate coordinates of center
                            cx_scaled, cy_scaled = annotator.estimate_coordinate(
                                c,
                                center_ref[0],
                                (unit_x, unit_y),
                            )
                            # Draw coordinate label
                            im0 = annotator.draw_coordinate_label(
                                im0,
                                c,
                                (cx_scaled, cy_scaled),
                                text=f'[{round(cx_scaled,2)}, {round(cy_scaled,2)}]',
                                font=cv2.FONT_HERSHEY_COMPLEX,
                                font_scale=0.5,
                                thickness=1,
                                color=obj_color[label],
                            )
                            
                            if opt.save_csv and label == "object":
                                dict_det["path"].append(p.name)
                                dict_det["x_coor"].append(round(cx_scaled,2))
                                dict_det["y_coor"].append(round(cy_scaled,2))

                    # Show table size
                    if label == "table":
                        bbox = measurement["xyxy"][label][0]
                        box_w, box_h = annotator.estimate_box_size(bbox, (unit_x, unit_y))
                        # show width size
                        im0 = cv2.putText(
                            im0,
                            f'TABLE SIZE: (WIDTH={round(box_w, 2)}, HEIGHT={round(box_h, 2)})', 
                            (
                                int(bbox[0] + 200),
                                int(bbox[1] - 2)
                            ),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                            fontScale=1,
                            color=obj_color[label],
                            thickness=opt.line_thickness,
                            lineType=cv2.LINE_AA,
                        )
                    
                # Save results (image with detection)
                cv2.imwrite(save_path, im0)
                print(f"Image written in the {save_path}")

    # Save .csv
    if opt.save_csv:
        csv_path = str(save_dir / "results.csv")
        df = pd.DataFrame(dict_det)
        df.to_csv(csv_path, index=False)
        print(f"CSV file written in the {csv_path}")

    print("Done.")

