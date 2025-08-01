# 統一集中所有「固定參數」或「路徑參數」
SCAP_MODEL_PATH = "models/fasterrcnn_scaphoid_best.pth"
FRACT_MODEL1_PATH = "models/5d/miracle_best_model_325.pth"
FRACT_MODEL2_PATH = "models/5d/model_5d_resnet_iou_best.pth"
FRACT_MODEL3_PATH = "models/5d/model_5d_resnet_detc_best.pth"

SCAP_ANNO_FOLDER = "test/scap_anno"
FRAC_ANNO_FOLDER = "test/frac_anno"
TEST_SCAP_IMG_FOLDER = "test/scap_img"
TEST_FRAC_IMG_FOLDER = "test/frac_img"

# 預測時的信心門檻
SCAP_CONFIDENCE = 0.5

from shapely.geometry import Polygon
import torch
import cv2
from torchvision.transforms import functional as F
# import faster_rcnnv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from model5d import ResNet50RotatedBBoxNet_5d
import torchvision.transforms as T
import numpy as np
from PIL import Image
import json
import os


common_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

def predict_scap_gt(annotation_filepath, img):
    """
    根據提供的標註 JSON 檔案，在圖片上繪製紅色的邊界框，並返回繪製後的圖片。

    參數:
        annotation_filepath (str): 包含邊界框資訊的標註 JSON 檔案路徑
        img (numpy.ndarray): 要繪製邊界框的圖片 (BGR 格式的 NumPy 陣列)

    返回:
        numpy.ndarray: 繪製了邊界框的圖片。
    """
    # 讀取 JSON 檔案
    try:
        with open(annotation_filepath, 'r') as f:
            annotation = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"無法找到指定的 JSON 檔案：{annotation_filepath}")
    except json.JSONDecodeError:
        raise ValueError("提供的 JSON 檔案無法解析為有效的格式。")

    for item in annotation:
        if "bbox" in item:
            bbox = [int(coord) for coord in item["bbox"]]
            x_min, y_min, x_max, y_max = bbox

            # 繪製紅色邊界框
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    return img


def predict_scap(model, confidence, img):
    """
    使用 Faster R-CNN 模型對圖片進行預測，並返回畫框後的圖片、裁切的圖片和邊界框座標。

    參數:
        model (torch.nn.Module): 已載入的 Faster R-CNN 模型
        confidence (float): 信心程度的閾值
        img (numpy.ndarray): 要預測的圖片 (BGR 格式的 NumPy 陣列)

    返回:
        tuple: (畫框後圖片, 裁切後的圖片列表, 邊界框座標列表)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 轉換圖片格式並轉換為 tensor
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).to(device)

    # 禁用梯度計算
    with torch.no_grad():
        predictions = model([image_tensor])[0]

    pred_boxes = predictions['boxes'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()

    high_conf_indices = pred_scores >= confidence
    filtered_boxes = pred_boxes[high_conf_indices]

    cropped_images = []
    bbox_coords = []
    img_no_bbox = img.copy()

    for box in filtered_boxes:
        x_min, y_min, x_max, y_max = box.astype(int)

        cropped_img = img_no_bbox[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped_img)
        bbox_coords.append((x_min, y_min, x_max, y_max))

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return img, cropped_images, bbox_coords


def predict_frac_det(model, img):
    """
    使用單一模型對圖片進行骨折檢測。

    參數：
        model: 已訓練的PyTorch模型
        img: 輸入的圖片，可以是PIL Image、numpy.ndarray 或圖片路徑

    返回：
        pred: 預測結果 (1 表示 '有骨折', 0 表示 '無骨折')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img_tensor = common_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_cls, _ = model(img_tensor)
        pred = torch.argmax(logits_cls, dim=1).item()

    return pred


def predict_frac_det_mix(model1, model2, model3, img):
    """
    使用三個模型對圖片進行骨折檢測，並通過多數投票決定最終結果。

    參數：
        model1, model2, model3: 已訓練的PyTorch模型
        img: 輸入的圖片，可以是PIL Image、numpy.ndarray 或圖片路徑

    返回：
        '有骨折' 或 '無骨折'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [model1, model2, model3]
    for model in models:
        model.to(device)
        model.eval()

    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img_tensor = common_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = []
        for model in models:
            logits_cls, _ = model(img_tensor)
            pred = torch.argmax(logits_cls, dim=1).item()
            preds.append(pred)

    final_pred = 1 if sum(preds) >= 2 else 0
    return '有骨折' if final_pred == 1 else '無骨折'


def predict_frac_gt(anno_name, img):
    """
    根據指定的標註名稱和圖像，繪製 frac 的旋轉框並返回相關信息。

    參數:
    - anno_name (str): 標註文件的名稱（不包含路徑）
    - img (numpy.ndarray): 原始圖像（BGR格式）

    返回:
    - img_with_bboxes (numpy.ndarray): 繪製了旋轉框的圖像
    - label (int): 標籤，1 表示存在 frac 標註，0 表示不存在
    - frac_real_coords (list or None): 真實的 frac bbox 座標，如果不存在 frac 標註則為 None
    """
    scap_anno_path = os.path.join('test', 'scap_anno', anno_name)
    frac_anno_path = os.path.join('test', 'frac_anno', anno_name)

    img_with_bboxes = img.copy()

    if not os.path.exists(scap_anno_path):
        print(f"Scap annotation file {scap_anno_path} does not exist.")
        return img, 0, None

    with open(scap_anno_path, 'r') as f:
        scap_data = json.load(f)

    if not scap_data or 'bbox' not in scap_data[0]:
        print(f"No scap bbox found in {scap_anno_path}.")
        return img, 0, None

    scap_bbox = scap_data[0]['bbox']
    scap_x, scap_y = scap_bbox[:2]

    if not os.path.exists(frac_anno_path):
        print(f"Frac annotation file {frac_anno_path} does not exist.")
        return img_with_bboxes, 0, None

    with open(frac_anno_path, 'r') as f:
        frac_data = json.load(f)

    if not frac_data or not frac_data[0].get('bbox'):
        return img_with_bboxes, 0, None

    frac_bbox = frac_data[0]['bbox']
    label = 1

    if len(frac_bbox) != 4:
        print(f"Invalid frac bbox in {frac_anno_path}.")
        return img_with_bboxes, 0, None

    frac_real_coords = [
        (int(scap_x + point[0]), int(scap_y + point[1])) for point in frac_bbox
    ]

    for i in range(4):
        pt1 = frac_real_coords[i]
        pt2 = frac_real_coords[(i + 1) % 4]
        cv2.line(img_with_bboxes, pt1, pt2, (0, 0, 255), 2)

    return img_with_bboxes, label, frac_real_coords


import math

def order_points_clockwise(pts):
    """
    將四個點按順時針方向排序為左上、右上、右下、左下。
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    ordered = [tuple(tl), tuple(tr), tuple(br), tuple(bl)]
    return ordered


def cxcywh_angle_to_four_points(cx, cy, w, h, angle):
    """
    將中心點、寬、高和角度轉換為四個旋轉框頂點。
    """
    rot_mat = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)]
    ])

    dx = w / 2
    dy = h / 2
    corners = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy]
    ])

    corners = np.dot(corners, rot_mat.T)
    corners += np.array([cx, cy])

    return corners


def predict_frac_bbox(model, img_whole, img_frac, scap_anno):
    """
    根據 scapula 的標註，預測 frac 的邊界框，並在圖像上繪製旋轉框。
    """
    if not scap_anno or not isinstance(scap_anno, list) or not all(len(box) == 4 for box in scap_anno):
        raise ValueError("scap_anno 必須是一個包含一個元組的列表，每個元組有四個數值 (x1, y1, x2, y2)。")

    scap_box = scap_anno[0]
    scap_x1, scap_y1, scap_x2, scap_y2 = scap_box
    scap_width = scap_x2 - scap_x1
    scap_height = scap_y2 - scap_y1

    common_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    img_frac_rgb = cv2.cvtColor(img_frac, cv2.COLOR_BGR2RGB)
    pil_img_frac = Image.fromarray(img_frac_rgb)
    transformed_img_frac = common_transform(pil_img_frac).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transformed_img_frac = transformed_img_frac.to(device)

    with torch.no_grad():
        output = model(transformed_img_frac)
        if isinstance(output, tuple):
            _, labels_5d = output
        else:
            labels_5d = output

        if not isinstance(labels_5d, torch.Tensor):
            raise TypeError("模型輸出 labels_5d 類型錯誤，預期為 torch.Tensor")

        labels_5d = labels_5d[0].cpu().numpy()
        if len(labels_5d) != 5:
            raise ValueError(f"labels_5d 的長度不正確，預期為 5，實際為 {len(labels_5d)}")

        cx, cy, w, h, angle = labels_5d
        four_pts = cxcywh_angle_to_four_points(cx, cy, w, h, angle)

        orig_height, orig_width = img_frac.shape[:2]
        resized_height, resized_width = 256, 256

        scale_x = orig_width / resized_width
        scale_y = orig_height / resized_height

        frac_bbox_scaled = [(pt[0] * scale_x, pt[1] * scale_y) for pt in four_pts]
        frac_real_coords = [(scap_x1 + p[0], scap_y1 + p[1]) for p in frac_bbox_scaled]

        frac_real_coords_ordered = order_points_clockwise(frac_real_coords)

        img_with_bbox = img_whole.copy()
        for i in range(4):
            pt1 = tuple(map(int, frac_real_coords_ordered[i]))
            pt2 = tuple(map(int, frac_real_coords_ordered[(i + 1) % 4]))
            cv2.line(img_with_bbox, pt1, pt2, (0, 0, 255), 2)

    return img_with_bbox, frac_real_coords_ordered




def calculate_iou(box1, box2):
    """
    計算兩個四邊形之間的 IoU（交並比）。
    """
    box1 = np.array(box1).astype(np.float32)
    box2 = np.array(box2).astype(np.float32)

    retval, intersection = cv2.intersectConvexConvex(box1, box2)
    if retval == 0:
        intersection_area = 0.0
    else:
        intersection_area = cv2.contourArea(intersection)

    area1 = cv2.contourArea(box1)
    area2 = cv2.contourArea(box2)

    iou = intersection_area / (area1 + area2 - intersection_area) if (area1 + area2 - intersection_area) != 0 else 0
    return iou
