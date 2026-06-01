import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
import torch
from torch import Tensor
from torchvision.ops.boxes import box_area
from typing import Tuple, List, Union
import math
import re
import base64
import io
import random
from PIL import Image, ImageDraw

def convert_to_qwen2vl_format(bbox, h, w):
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 / w * 1000)
    y1_new = round(y1 / h * 1000)
    x2_new = round(x2 / w * 1000)
    y2_new = round(y2 / h * 1000)
    
    x1_new = max(0, min(x1_new, 1000))
    y1_new = max(0, min(y1_new, 1000))
    x2_new = max(0, min(x2_new, 1000))
    y2_new = max(0, min(y2_new, 1000))
    
    return [x1_new, y1_new, x2_new, y2_new]

def convert_from_qwen2vl_format(bbox_new, h, w):
    """
    将Qwen2-VL格式的边界框坐标转换回原始图像坐标
    
    参数:
        bbox_new: Qwen2-VL格式的边界框 [x1_new, y1_new, x2_new, y2_new]
        h: 原始图像高度
        w: 原始图像宽度
    
    返回:
        原始边界框坐标 [x1, y1, x2, y2]
    """
    x1_new, y1_new, x2_new, y2_new = bbox_new
    
    # 逆转换：将归一化坐标映射回原始图像尺度
    x1 = x1_new * w / 1000.0
    y1 = y1_new * h / 1000.0
    x2 = x2_new * w / 1000.0
    y2 = y2_new * h / 1000.0
    
    # 确保坐标在合理范围内
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    return [x1, y1, x2, y2]

def extract_bounding_boxes(data_list):
    """
    从数据列表中提取segment_id与bounding box的映射关系。
    
    参数:
        data_list: 列表，每个元素包含segment_id和output_text字段
        
    返回:
        dict: 键为segment_id，值为bounding box列表（如[x_min, y_min, x_max, y_max]）
    """
    segment_to_bbox = {}
    # 定义可能的bounding box字段名（覆盖数据中的变体）
    possible_bbox_keys = ['bbox', 'bounding_box', 'bbox_2d', "bounding box", "bounding_box_2d"]
    
    for item in data_list:
        segment_id = item.get('segment_id')
        # 使用output_text字段（如果output_text_special更准确，可替换为item['output_text_special'][0]）
        output_text = item['output_text'][0]  
        
        # 清理JSON字符串：移除代码块标记和特殊符号
        if output_text.startswith('```json'):
            json_str = output_text[7:-3].strip()  # 移除```json和尾部的```
        else:
            json_str = output_text
        if '<|im_end|>' in json_str:
            json_str = json_str.replace('<|im_end|>', '').strip()
        
        try:
            parsed_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"解析错误 segment_id {segment_id}: {e}")
            continue
        
        bbox = None
        # 情况1: parsed_data是字典（如{"person": {"bbox": [...]}}）
        if isinstance(parsed_data, dict):
            # 先检查顶层键
            for key in possible_bbox_keys:
                if key in parsed_data:
                    bbox = parsed_data[key]
                    break
            # 若顶层未找到，检查嵌套结构
            if bbox is None:
                for value in parsed_data.values():
                    if isinstance(value, dict):
                        for key in possible_bbox_keys:
                            if key in value:
                                bbox = value[key]
                                break
                    if bbox is not None:
                        break
        # 情况2: parsed_data是列表（如[{"bbox_2d": [...]}]）
        elif isinstance(parsed_data, list):
            for obj in parsed_data:
                if isinstance(obj, dict):
                    for key in possible_bbox_keys:
                        if key in obj:
                            bbox = obj[key]
                            break
                if bbox is not None:
                    break
        
        # 验证bounding box格式（应为4个数值的列表）
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(i, (int, float)) for i in bbox):
            segment_to_bbox[segment_id] = bbox
        else:
            print(f"未找到有效bounding box: segment_id {segment_id}")
    
    return segment_to_bbox


def show_mask(mask, ax, random_color=False, borders = True):
    '''
    在指定的坐标轴上可视化分割掩码。
    
    此函数将二值掩码转换为带透明度的彩色图像，并可选择添加边界轮廓以增强可视化效果。
    主要用于图像分割任务的结果展示。

    Args:
        mask (numpy.ndarray): 二值掩码数组，形状为(H, W)或(1, H, W)，值为0或1。
        ax (matplotlib.axes.Axes): matplotlib坐标轴对象，用于绘制掩码。
        random_color (bool, optional): 是否使用随机颜色，默认为False（使用预定义的蓝色）。
        borders (bool, optional): 是否在掩码边界添加轮廓线，默认为True。

    Returns:
        None: 函数直接修改传入的坐标轴对象，不返回任何值。
    '''
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    '''
    在图像上显示正负点标注。
    
    此函数根据点的标签（正/负）在指定坐标轴上用不同颜色的星形标记显示点坐标。
    常用于交互式分割任务的提示点可视化。

    Args:
        coords (numpy.ndarray): 点坐标数组，形状为(N, 2)，每行表示一个点的(x, y)坐标。
        labels (numpy.ndarray): 点标签数组，形状为(N,)，值为1表示正点，0表示负点。
        ax (matplotlib.axes.Axes): matplotlib坐标轴对象，用于绘制点。
        marker_size (int, optional): 标记点的大小，默认为375。

    Returns:
        None: 函数直接修改传入的坐标轴对象，不返回任何值。
    '''
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    '''
    在图像上显示边界框。
    
    此函数在指定坐标轴上绘制一个绿色矩形框，表示目标检测或分割中的边界框。

    Args:
        box (list or tuple): 边界框坐标，格式为[x0, y0, x1, y1]（左上角和右下角坐标）。
        ax (matplotlib.axes.Axes): matplotlib坐标轴对象，用于绘制边界框。

    Returns:
        None: 函数直接修改传入的坐标轴对象，不返回任何值。
    '''
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True,name=""):
    '''
    综合可视化分割结果，包括原图像、掩码、得分及可能的交互提示。
    
    此函数生成完整的可视化图像，显示分割掩码、置信度分数，并可选择性地显示点提示和框提示。
    适用于评估和展示分割模型的输出结果。

    Args:
        image (numpy.ndarray): 原始输入图像数组，形状为(H, W, 3)。
        masks (list of numpy.ndarray): 分割掩码列表，每个掩码为二值数组。
        scores (list of float): 每个掩码对应的置信度分数列表。
        point_coords (numpy.ndarray, optional): 点提示坐标数组，形状为(N, 2)。
        box_coords (list or tuple, optional): 框提示坐标，格式为[x0, y0, x1, y1]。
        input_labels (numpy.ndarray, optional): 点提示对应的标签数组（1为正点，0为负点）。
        borders (bool, optional): 是否在掩码边界添加轮廓线，默认为True。
        name (str, optional): 保存结果图像的文件路径，默认为空字符串（不保存）。

    Returns:
        None: 函数生成并显示图像，或保存到指定路径，不返回任何值。
    '''
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        # os.makedirs("demo_dir",exist_ok=True)
        plt.savefig(name)
        plt.close()



def compute_iou(box1, box2):
    """
    计算两个边界框之间的普通IOU（交并比）
    
    参数:
        box1: list of 4 numbers [x1, y1, x2, y2] (左上角和右下角坐标)
        box2: list of 4 numbers [x1, y1, x2, y2] (左上角和右下角坐标)
    
    返回:
        iou: float, 交并比的值
    """
    # 提取box1和box2的坐标
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集区域的坐标
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    # 计算交集区域的面积
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    
    # 计算两个边界框各自的面积
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集面积
    union_area = area_box1 + area_box2 - inter_area
    
    # 避免除以零的情况
    if union_area == 0:
        return 0.0
    
    # 计算IOU
    iou = inter_area / union_area
    return iou

def compute_ciou(box1, box2):
    """
    计算两个边界框之间的CIOU（Complete IoU）
    
    参数:
        box1: list of 4 numbers [x1, y1, x2, y2] (左上角和右下角坐标)
        box2: list of 4 numbers [x1, y1, x2, y2] (左上角和右下角坐标)
    
    返回:
        ciou: float, CIOU的值
    """
    # 首先计算普通IOU
    iou = compute_iou(box1, box2)
    
    # 提取box1和box2的坐标
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算两个框的中心点坐标
    center_x1 = (x1_1 + x2_1) / 2.0
    center_y1 = (y1_1 + y2_1) / 2.0
    center_x2 = (x1_2 + x2_2) / 2.0
    center_y2 = (y1_2 + y2_2) / 2.0
    
    # 计算中心点之间的欧氏距离的平方
    center_distance_squared = (center_x1 - center_x2)**2 + (center_y1 - center_y2)**2
    
    # 计算最小外接矩形的坐标（能够同时包含两个框的最小矩形）
    enclose_x1 = min(x1_1, x1_2)
    enclose_y1 = min(y1_1, y1_2)
    enclose_x2 = max(x2_1, x2_2)
    enclose_y2 = max(y2_1, y2_2)
    
    # 计算最小外接矩形的对角线距离的平方
    enclose_diagonal_squared = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2
    
    # 计算两个框的宽和高
    w1 = x2_1 - x1_1
    h1 = y2_1 - y1_1
    w2 = x2_2 - x1_2
    h2 = y2_2 - y1_2
    
    # 计算宽高比的相似性参数v
    v = (4 / (math.pi ** 2)) * (math.atan(w2 / (h2 + 1e-10)) - math.atan(w1 / (h1 + 1e-10))) ** 2
    
    # 计算CIOU的权重参数alpha
    alpha = v / (1 - iou + v + 1e-10)
    
    # 计算CIOU
    ciou = iou - (center_distance_squared / (enclose_diagonal_squared + 1e-10)) - (alpha * v)
    
    return ciou

def mask2xyxy(mask_np):
    nonzero_indices = np.where(mask_np > 0)
    # 计算物体的边界框
    y_min, y_max = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    x_min, x_max = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    return [x_min.tolist(), y_min.tolist(), x_max.tolist(), y_max.tolist()]


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def find_points(bbox, image, judge_point_in_object, internal_count=3, external_count=2):
    """
    找出指定数量的物体内部点和外部点，尝试次数较少。
    
    参数:
        bbox: 元组 (x_min, y_min, x_max, y_max)，表示边界框。
        image: 传递给 judge_point_in_object 的图像对象。
        judge_point_in_object: 函数，接收点坐标和图像，返回True（点在物体内）或False。
        internal_count: 需要找的物体内部点数量，默认为3。
        external_count: 需要找的物体外部点数量，默认为2。
        
    返回:
        internal_points: 物体内部点的列表，每个点为元组 (x, y)。
        external_points: 物体外部点的列表，每个点为元组 (x, y)。
    """
    internal_points = []
    external_points = []
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    # 1. 优先采样角点找外部点（冗余bbox下角点大概率外部）
    corners = [
        (x_min, y_min),  # 左下角
        (x_max, y_min),  # 右下角
        (x_min, y_max),  # 左上角
        (x_max, y_max)   # 右上角
    ]
    for corner in corners:
        if len(external_points) >= external_count:
            break
        if not judge_point_in_object(corner, image):
            external_points.append(corner)
    
    # 2. 采样中心点找内部点（物体通常靠近中心）
    center = (x_min + width / 2, y_min + height / 2)
    if judge_point_in_object(center, image) and len(internal_points) < internal_count:
        internal_points.append(center)
    
    # 3. 基于中心点状态自适应搜索内部点
    if internal_points:
        # 中心点内部时，附近点高概率内部
        offsets = [(0, height/4), (0, -height/4), (width/4, 0), (-width/4, 0)]
        for dx, dy in offsets:
            if len(internal_points) >= internal_count:
                break
            point = (center[0] + dx, center[1] + dy)
            if judge_point_in_object(point, image):
                internal_points.append(point)
    else:
        # 中心点外部时，采样四分点（覆盖中心区域）
        quarter_points = [
            (x_min + width/4, y_min + height/4),
            (x_max - width/4, y_min + height/4),
            (x_min + width/4, y_max - height/4),
            (x_max - width/4, y_max - height/4)
        ]
        for point in quarter_points:
            if len(internal_points) >= internal_count:
                break
            if judge_point_in_object(point, image):
                internal_points.append(point)
    
    # 4. 后备采样确保点数足够
    # 内部点不足时，采样边界中点
    if len(internal_points) < internal_count:
        mid_points = [
            (x_min + width/2, y_min),  # 下中点
            (x_min + width/2, y_max),  # 上中点
            (x_min, y_min + height/2),  # 左中点
            (x_max, y_min + height/2)   # 右中点
        ]
        for point in mid_points:
            if len(internal_points) >= internal_count:
                break
            if judge_point_in_object(point, image):
                internal_points.append(point)
    
    # 外部点不足时，采样边界中点（角点可能意外内部）
    if len(external_points) < external_count:
        mid_points = [
            (x_min + width/2, y_min),
            (x_min + width/2, y_max),
            (x_min, y_min + height/2),
            (x_max, y_min + height/2)
        ]
        for point in mid_points:
            if len(external_points) >= external_count:
                break
            if not judge_point_in_object(point, image):
                external_points.append(point)
    
    return internal_points[:internal_count], external_points[:external_count]



def extract_single_bounding_box(text):
    """
    从JSON文本中提取边界框坐标
    
    参数:
        text: 包含JSON对象的文本，可能包含代码块标记
        
    返回:
        list: 包含所有找到的边界框坐标的列表，每个边界框是一个包含4个数字的列表
    """
    # 定义可能的边界框字段名称
    bbox_fields = ['bbox', 'bounding_box', 'bbox_2d', 'bounding box', 'bounding_box_2d']
    
    # 用于存储所有找到的边界框
    all_bboxes = []
    
    # 使用正则表达式提取JSON代码块
    json_blocks = re.findall(r'```json\n(.*?)\n```', text, re.DOTALL)
    
    # 如果没有找到代码块，尝试直接解析整个文本
    if not json_blocks:
        json_blocks = [text]
    
    for json_block in json_blocks:
        try:
            # 解析JSON数据
            data = json.loads(json_block.strip())
            
            # 递归搜索边界框字段
            def find_bboxes(obj, path=""):
                bboxes = []
                
                if isinstance(obj, dict):
                    # 检查当前字典中是否有边界框字段
                    for field in bbox_fields:
                        if field in obj:
                            bbox_value = obj[field]
                            if (isinstance(bbox_value, list) and 
                                len(bbox_value) == 4 and 
                                all(isinstance(x, (int, float)) for x in bbox_value)):
                                bboxes.append(bbox_value)
                    
                    # 递归搜索嵌套字典
                    for key, value in obj.items():
                        bboxes.extend(find_bboxes(value, f"{path}.{key}" if path else key))
                
                elif isinstance(obj, list):
                    # 递归搜索列表中的元素
                    for i, item in enumerate(obj):
                        bboxes.extend(find_bboxes(item, f"{path}[{i}]"))
                
                return bboxes
            
            # 在当前JSON块中查找边界框
            bboxes_in_block = find_bboxes(data)
            all_bboxes.extend(bboxes_in_block)
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            continue
    
    return all_bboxes

def get_radius_growth(t_normalized, growth_type='linear', power=1.0):
    """
    参数化半径增长函数
    
    参数:
    t_normalized: 归一化的参数 [0, 1]
    growth_type: 增长类型 ('linear', 'quadratic', 'sqrt', 'exponential', 'logarithmic', 'sigmoid')
    power: 幂函数的指数（当growth_type='power'时使用）
    """
    if growth_type == 'linear':
        return t_normalized
    elif growth_type == 'quadratic':
        return t_normalized**2
    elif growth_type == 'sqrt':
        return np.sqrt(t_normalized)
    elif growth_type == 'exponential':
        return (np.exp(3 * t_normalized) - 1) / (np.exp(3) - 1)
    elif growth_type == 'logarithmic':
        return np.log(1 + 9 * t_normalized) / np.log(10)
    elif growth_type == 'sigmoid':
        return 1 / (1 + np.exp(-8 * (t_normalized - 0.5)))
    elif growth_type == 'power':
        return t_normalized**power
    else:
        return t_normalized  # 默认线性

def superellipse_spiral_advanced(bbox, num_turns=8, num_points=3000, exponent=4.0, 
                                direction='CCW', end_point='right',growth_type="sqrt"):
    """
    增强版的超椭圆螺旋线，支持方向控制和终点位置控制
    
    参数:
    bbox: [x_min, y_min, x_max, y_max]
    num_turns: 螺旋线的圈数
    num_points: 生成的点数
    exponent: 超椭圆指数 (n=2是椭圆，n>2更接近矩形)
    direction: 螺旋方向 ('CW'顺时针 或 'CCW'逆时针)
    end_point: 终点位置 ('left', 'right', 'top', 'bottom', 或具体角度)
    """
    x_min, y_min, x_max, y_max = bbox
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    width, height = x_max - x_min, y_max - y_min
    
    # 生成参数
    theta = np.linspace(0, num_turns * 2 * np.pi, num_points)
    
    # 控制螺旋方向
    if direction == 'CW':  # 顺时针
        theta = -theta  # 反转角度
    
    # 控制终点位置
    end_point_angle_map = {
        'right': 0,          # 0度 (正右)
        'top': np.pi/2,      # 90度 (正上)
        'left': np.pi,       # 180度 (正左)
        'bottom': 3*np.pi/2  # 270度 (正下)
    }
    
    if end_point in end_point_angle_map:
        # 计算旋转角度，使终点位于指定位置
        end_angle = end_point_angle_map[end_point]
        # 当前终点角度是最后一个theta值
        current_end_angle = theta[-1] % (2*np.pi)
        rotation_angle = end_angle - current_end_angle
        theta += rotation_angle
    elif isinstance(end_point, (int, float)):
        # 如果end_point是数字，直接作为角度（弧度）
        current_end_angle = theta[-1] % (2*np.pi)
        rotation_angle = end_point - current_end_angle
        theta += rotation_angle
    
    # 超椭圆参数
    a, b = width / 2, height / 2  # 半长轴和半短轴
    
    # 半径线性增长
    # r = np.linspace(0, 1, len(theta))
    t_normalized = np.linspace(0, 1, len(theta))
    r = get_radius_growth(t_normalized, growth_type=growth_type)
    
    # 生成超椭圆螺旋线
    points = []
    for i, t in enumerate(theta):
        # 普通椭圆坐标
        x_ellipse = a * r[i] * np.cos(t)
        y_ellipse = b * r[i] * np.sin(t)
        
        # 应用超椭圆变换
        if x_ellipse != 0 and y_ellipse != 0:
            # 计算超椭圆上的点
            x_norm = abs(x_ellipse / a)
            y_norm = abs(y_ellipse / b)
            
            # 计算当前点的"超椭圆范数"
            current_norm = (x_norm**exponent + y_norm**exponent) ** (1/exponent)
            
            # 将点映射到超椭圆上
            if current_norm > 0:
                scale = r[i] / current_norm
                x_super = x_ellipse * scale
                y_super = y_ellipse * scale
            else:
                x_super, y_super = 0, 0
        else:
            x_super, y_super = x_ellipse, y_ellipse
        
        # 平移到矩形中心
        x_rect = cx + x_super
        y_rect = cy + y_super
        
        points.append((x_rect, y_rect))
    
    return np.array(points)

def sample_spiral_points(spiral_points, num_samples=20, method='uniform', 
                         randomness=0.2, random_seed=None, step=None,
                         dynamic_step=True, dynamic_range=(0.5, 1.5)):
    """
    增强版采样函数：固定基准步长 + 起点偏置 + 贴合螺旋密度的动态步长
    核心：步长随螺旋径向增长速率动态调整，匹配自然密度
    
    新增参数：
    dynamic_step: 是否启用动态步长（默认True）
    dynamic_range: 动态步长系数范围（默认(0.5,1.5)），避免步长过大/过小
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if method == 'uniform':
        indices = np.linspace(0, len(spiral_points)-1, num_samples, dtype=int)
        return spiral_points[indices]
    
    elif method == 'arc_length':
        # 1. 基础计算：累积弧长、螺旋中心、总长度
        diffs = np.diff(spiral_points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative_length = np.concatenate(([0], np.cumsum(segment_lengths)))
        total_length = cumulative_length[-1]
        
        # 计算螺旋中心（用于反推径向距离r）
        cx = (spiral_points[:,0].min() + spiral_points[:,0].max()) / 2
        cy = (spiral_points[:,1].min() + spiral_points[:,1].max()) / 2
        # 每个点到中心的径向距离（近似r的相对值）
        radial_dist = np.sqrt((spiral_points[:,0] - cx)**2 + (spiral_points[:,1] - cy)**2)
        # 归一化径向距离到[0,1]，匹配t_normalized的范围
        radial_dist_norm = (radial_dist - radial_dist.min()) / (radial_dist.max() - radial_dist.min() + 1e-10)
        
        # 2. 计算动态步长系数（核心：贴合径向增长密度）
        if dynamic_step and step is not None:
            # 计算径向增长速率（用径向距离的导数近似dr/dt）
            dr_dt = np.gradient(radial_dist_norm)
            # 归一化增长速率到[0,1]，避免极端值
            dr_dt_norm = (dr_dt - dr_dt.min()) / (dr_dt.max() - dr_dt.min() + 1e-10)
            # 映射到用户指定的动态范围（默认0.5~1.5）
            min_coeff, max_coeff = dynamic_range
            dynamic_coeff = min_coeff + (max_coeff - min_coeff) * dr_dt_norm
        else:
            # 不启用动态步长，系数恒为1（固定基准步长）
            dynamic_coeff = np.ones_like(radial_dist_norm)
        
        # 3. 生成基准采样距离（起点偏置 + 动态步长）
        base_distances = []
        if step is not None and step > 0:
            # 起点随机偏置b ∈ [0, step)
            b = np.random.uniform(0, step) if step > 0 else 0.0
            current_dist = b
            # 循环生成动态步长的采样点
            while current_dist <= total_length:
                base_distances.append(current_dist)
                # 找到当前距离对应的螺旋点索引，获取对应的动态系数
                idx = np.searchsorted(cumulative_length, current_dist)
                idx = min(idx, len(dynamic_coeff)-1)  # 避免越界
                # 实际步长 = 基准step × 动态系数
                actual_step = step * dynamic_coeff[idx]
                # 下一个采样点距离（确保步长为正）
                current_dist += max(actual_step, step * 0.1)  # 防止步长过小
            
            base_distances = np.array(base_distances)
            # 处理极端情况：确保至少2个采样点（起点+终点）
            if len(base_distances) < 2:
                base_distances = np.array([b, total_length])
        else:
            # 无step时，沿用原num_samples均匀采样（无偏置、无动态步长）
            base_distances = np.linspace(0, total_length, num_samples)
        
        # 4. 保留有约束的随机扰动（叠加在动态步长上）
        if randomness > 0:
            # 扰动幅度：基于基准step（有step时）或总长度比例（无step时）
            max_disturbance = step * randomness if (step is not None and step > 0) else \
                              total_length * randomness / len(base_distances)
            
            disturbances = np.random.uniform(-max_disturbance, max_disturbance, len(base_distances))
            # 边界约束：不超出螺旋线范围
            disturbances[0] = max(disturbances[0], -base_distances[0])
            disturbances[-1] = min(disturbances[-1], total_length - base_distances[-1])
            # 扰动后强制递增（保证采样方向不变）
            disturbed_distances = np.sort(base_distances + disturbances)
            disturbed_distances[0] = max(disturbed_distances[0], 0)
            disturbed_distances[-1] = min(disturbed_distances[-1], total_length)
        else:
            disturbed_distances = base_distances
        
        # 5. 插值生成采样点
        interp_x = np.interp(disturbed_distances, cumulative_length, spiral_points[:, 0])
        interp_y = np.interp(disturbed_distances, cumulative_length, spiral_points[:, 1])
        
        return np.column_stack((interp_x, interp_y))
    
    elif method == 'curvature':
        dx = np.gradient(spiral_points[:, 0])
        dy = np.gradient(spiral_points[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10)**1.5
        curvature_normalized = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature) + 1e-10)
        weights = curvature_normalized / np.sum(curvature_normalized)
        indices = np.random.choice(len(spiral_points), size=num_samples, replace=False, p=weights)
        indices.sort()
        return spiral_points[indices]
    
    else:
        indices = np.linspace(0, len(spiral_points)-1, num_samples, dtype=int)
        return spiral_points[indices]
    


# ---------------------- 新增：随机参数生成函数（核心修改） ----------------------
def get_random_spiral_params():
    """
    随机生成螺旋线参数：方向 + 两个不同的端点组合
    返回：direction（CW/CCW）、combine（(端点1, 端点2)）
    """
    # 所有可选端点
    all_end_points = ['top', 'bottom', 'left', 'right']
    # 随机选择2个不同的端点（无重复）
    combine = random.sample(all_end_points, 2)  # 结果如 ('top', 'left')、('right', 'bottom') 等
    # 随机选择方向
    direction = random.choice(["CW", "CCW"])
    # return "CW", ['top','top']
    return direction, tuple(combine)

def generate_spiral_candidate_points(bbox, num_samples, 
                                     num_turns=8, num_points=3000, exponent=5.0,
                                     growth_type="linear", sample_method="arc_length",
                                     dynamic_step = 0.8, step=50, randomness=0.2, random_seed=None):
    """
    合并版函数：从同一条螺旋曲线生成 exter（最前面）和 inter（最后面）两组点
    
    参数说明（沿用你的配置，新增统一控制参数）：
    bbox: 边界框 [xmin, ymin, xmax, ymax]
    num_samples: 每组采样点数（exter和inter各num_samples个）
    num_turns: 螺旋圈数（共享）
    num_points: 螺旋总点数（共享）
    exponent: 超椭圆指数（共享）
    growth_type: 径向增长类型（共享）
    sample_method: 采样方法（共享）
    step: 采样步长（共享你修改后的动态步长逻辑）
    randomness: 采样随机性（可选，适配你之前的动态步长+扰动）
    random_seed: 随机种子（可选，复现结果）
    """
    # 1. 生成随机参数（方向+端点组合，共享）
    direction, combine = get_random_spiral_params()
    
    # 2. 生成单条螺旋曲线（核心：exter和inter共享同一条曲线）
    spiral = superellipse_spiral_advanced(
        bbox, num_turns=num_turns, num_points=num_points, exponent=exponent,
        direction=direction, end_point=combine[0], growth_type=growth_type
    )
    
    # 3. 采样足够多的点（按你的逻辑：num_samples*num_turns，确保有足够点拆分）
    total_sampled = sample_spiral_points(
        spiral, num_samples=num_samples * num_turns,  # 足够多的采样点
        method=sample_method, dynamic_step=dynamic_step, step=step, 
        randomness=randomness, random_seed=random_seed
    )
    
    # 4. 拆分 exter 和 inter（严格保留你的逻辑）
    exter_points = total_sampled[:num_samples]  # 最前面的num_samples个点
    inter_points = total_sampled[::-1][:num_samples]  # 反转后取前（原曲线最后面）num_samples个点
    
    # 5. 返回结果（两组点+原始螺旋+参数，方便调试）
    return exter_points, inter_points, spiral, (direction, combine)

def generate_internal_candidate_points(bbox, count):
    step = max(bbox[3]-bbox[1],bbox[2]-bbox[0])
    dynamic_step = 0.8
    randomness=0.2

    _, inter, _, _ = generate_spiral_candidate_points(
        bbox=bbox, num_samples=count, num_turns=8, growth_type="sigmoid",
        sample_method="arc_length", step=step, randomness=randomness
    )
    
    points = [(float(x), float(y)) for x, y in inter]
    
    return points

def generate_external_candidate_points(bbox, count):
    step = max(bbox[3]-bbox[1],bbox[2]-bbox[0])
    dynamic_step = 0.8
    randomness=0.2

    exter, _, _, _ = generate_spiral_candidate_points(
        bbox=bbox, num_samples=count, num_turns=8, growth_type="sigmoid",
        sample_method="arc_length", step=step, randomness=randomness
    )
    
    points = [(float(x), float(y)) for x, y in exter]
    
    return points


def find_points_modular(bbox, image, judge_point_in_object, internal_count=3, external_count=2):
    """
    模块化版本：找出指定数量的物体内部点和外部点。

    参数:
        bbox: 元组 (x_min, y_min, x_max, y_max)，表示边界框。
        image: 传递给 judge_point_in_object 的图像对象。
        judge_point_in_object: 函数，接收点坐标和图像，返回True（点在物体内）或False。
        internal_count: 需要找的物体内部点数量，默认为3。
        external_count: 需要找的物体外部点数量，默认为2。

    返回:
        internal_points: 物体内部点的列表。
        external_points: 物体外部点的列表。
    """
    internal_points = []
    external_points = []

    # 生成候选点列表
    internal_candidates = generate_internal_candidate_points(bbox, internal_count)
    external_candidates = generate_external_candidate_points(bbox, external_count)

    # 遍历内部点候选列表，判断并收集内部点
    for point in internal_candidates:
        if len(internal_points) >= internal_count:
            break
        if judge_point_in_object(point, image):
            internal_points.append(point)

    # 遍历外部点候选列表，判断并收集外部点
    for point in external_candidates:
        if len(external_points) >= external_count:
            break
        if not judge_point_in_object(point, image):
            external_points.append(point)

    return internal_points[:internal_count], external_points[:external_count]


def draw_marker_on_image(image, p1, shape, size, bbox):
    """
    在图像上绘制标记点，确保裁剪区域包含原始物体和标记，并返回相关信息。

    参数:
        image: PIL Image对象
        p1: 原始点坐标 (x, y)
        shape: 图形类型 ('star', 'hexagon', 'circle')
        size: 图形大小（半径）
        bbox: 原始物体的边界框 (x_min, y_min, x_max, y_max)

    返回:
        tuple: (绘制后的图像, 颜色RGB, 颜色名称, 调整后位置, crop_bbox, size, 图形名称)
    """
    # 创建图像的副本，避免修改原图
    img_copy = image.copy()
    img_width, img_height = img_copy.size

    # 预定义显眼颜色列表 (RGB, 颜色名称)
    colors = [
        ((255, 0, 0), "red"),
        ((0, 255, 0), "green"),
        ((0, 0, 255), "blue"),
        ((255, 255, 0), "yellow"),
        ((255, 0, 255), "carmine"),
        ((0, 255, 255), "cyan"),
        ((255, 165, 0), "orange"),
        ((255, 192, 203), "pink")
    ]
    color, color_name = random.choice(colors)

    # 1. 计算能同时包含物体bbox和标记图形的理想裁剪区域
    # 物体bbox
    x_min_obj, y_min_obj, x_max_obj, y_max_obj = bbox
    # 标记图形的有效区域（考虑size）
    marker_left = p1[0] - size
    marker_right = p1[0] + size
    marker_top = p1[1] - size
    marker_bottom = p1[1] + size

    # 合并区域：取物体bbox和标记图形区域的并集
    ideal_crop_left = min(x_min_obj, marker_left)
    ideal_crop_top = min(y_min_obj, marker_top)
    ideal_crop_right = max(x_max_obj, marker_right)
    ideal_crop_bottom = max(y_max_obj, marker_bottom)

    # 2. 边界安全修正，确保裁剪框在图像范围内[3](@ref)
    # 计算修正量，确保不越界。如果越界，则调整标记绘制位置。
    padding = size * 2  # 在标记周围保留一些额外空间
    safe_crop_left = max(0, ideal_crop_left - padding)
    safe_crop_top = max(0, ideal_crop_top - padding)
    safe_crop_right = min(img_width, ideal_crop_right + padding)
    safe_crop_bottom = min(img_height, ideal_crop_bottom + padding)

    # 如果因为图像边界导致标记无法完整显示，调整标记位置
    adjusted_x = p1[0]
    adjusted_y = p1[1]
    # 检查标记的左侧和上侧是否会因裁剪框修正而被切掉
    if safe_crop_left > ideal_crop_left - padding:
        # 左边空间不足，标记右移
        adjusted_x = max(p1[0] + (safe_crop_left - (ideal_crop_left - padding)), size)
    if safe_crop_top > ideal_crop_top - padding:
        # 上边空间不足，标记下移
        adjusted_y = max(p1[1] + (safe_crop_top - (ideal_crop_top - padding)), size)
    # 检查标记的右侧和下侧是否会因裁剪框修正而被切掉
    if safe_crop_right < ideal_crop_right + padding:
        # 右边空间不足，标记左移（但不能小于size）
        adjusted_x = min(p1[0] - ((ideal_crop_right + padding) - safe_crop_right), img_width - size)
    if safe_crop_bottom < ideal_crop_bottom + padding:
        # 下边空间不足，标记上移（但不能小于size）
        adjusted_y = min(p1[1] - ((ideal_crop_bottom + padding) - safe_crop_bottom), img_height - size)

    adjusted_x = max(size, min(img_width - size, adjusted_x))
    adjusted_y = max(size, min(img_height - size, adjusted_y))
    adjusted_pos = (adjusted_x, adjusted_y)

    # 根据调整后的标记位置，重新计算最终的安全裁剪框
    final_crop_left = max(0, min(x_min_obj, adjusted_x - size) - padding)
    final_crop_top = max(0, min(y_min_obj, adjusted_y - size) - padding)
    final_crop_right = min(img_width, max(x_max_obj, adjusted_x + size) + padding)
    final_crop_bottom = min(img_height, max(y_max_obj, adjusted_y + size) + padding)
    crop_bbox = (final_crop_left, final_crop_top, final_crop_right, final_crop_bottom)

    # 创建绘图对象并绘制图形
    draw = ImageDraw.Draw(img_copy)
    if shape == 'circle':
        bbox_circle = (adjusted_x - size, adjusted_y - size, adjusted_x + size, adjusted_y + size)
        draw.ellipse(bbox_circle, fill=color, outline=(0, 0, 0), width=2)
    elif shape == 'star':
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            radius = size if i % 2 == 0 else size * 0.4
            x = adjusted_x + radius * math.cos(angle)
            y = adjusted_y - radius * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color, outline=(0, 0, 0))
    elif shape == 'hexagon':
        points = []
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3
            x = adjusted_x + size * math.cos(angle)
            y = adjusted_y - size * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color, outline=(0, 0, 0))

    return (img_copy, color, color_name, adjusted_pos, crop_bbox, size, shape)


# 假设img是已读取的PIL.Image对象
def image_to_base64(img, format='JPEG'):
    # 创建内存字节流
    img_byte_arr = io.BytesIO()
    # 将图像保存到流中，指定格式
    img.save(img_byte_arr, format=format)
    # 获取字节数据并编码
    img_byte_arr.seek(0)  # 重置流位置
    base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    return base64_image



def analyze_generation_logits(outputs, inputs, tokenizer):
    """分析生成过程中每一步的logits"""
    generated_ids = outputs.sequences
    logits_sequence = outputs.scores
    
    # 提取纯生成部分
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    analysis_results = []
    
    for step, (logits, token_id) in enumerate(zip(logits_sequence, generated_ids_trimmed[0])):
        # 当前步骤的logits
        step_logits = logits[0]  # 取batch中第一个样本
        step_probs = torch.softmax(step_logits, dim=-1)
        
        # 实际生成token的概率
        token_prob = step_probs[token_id].item()
        token_text = tokenizer.decode(token_id)
        
        # 获取top-k候选
        topk_probs, topk_indices = torch.topk(step_probs, k=5)
        
        # 收集分析结果
        step_result = {
            'step': step,
            'token': token_text,
            'token_id': token_id.item(),
            'probability': token_prob,
            'top_candidates': [
                (tokenizer.decode(idx.item()), prob.item()) 
                for prob, idx in zip(topk_probs, topk_indices)
            ]
        }
        analysis_results.append(step_result)
        
        # # 打印当前步骤信息
        # print(f"Step {step}: '{token_text}' (prob: {token_prob:.4f})")
        # print("Top alternatives:", [
        #     f"{text}({prob:.4f})" for text, prob in step_result['top_candidates']
        # ])
        # print("-" * 50)
    
    return analysis_results

def get_points_check_messages(base64_image,color_name,sentence):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64_image,
                },
                {"type": "text", "text": "Answer strictly yes or no: Is the {}-colored star on the object referred to by `{}' in the picture?".format(color_name, sentence)},
            ],
        }
    ]

def get_points_check_messages_corr(base64_image,coordinate,sentence):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64_image,
                },
                # {"type": "text", "text": "Answer strictly yes or no: Is the Point ({},{}) on the object referred to by `{}' in the picture?".format(coordinate[0],coordinate[1], sentence)},
                # {"type": "text", "text": "Answer strictly yes or no: Is the Point ({},{}) on the object referred to by `{}' in the picture?".format(coordinate[0],coordinate[1], sentence)},
                {"type": "text", "text": "Answer strictly yes or no: Is the Point x = {}, y = {} on the object referred to by `{}' in the picture?".format(coordinate[0],coordinate[1], sentence)},
            ],
        }
    ]

def inference_bbox(model, processor, tokenizer, img, sentence, task="refcoco"):
    with open(img, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8") 

    if task == "refcoco" or task == "refcoco+" or task == "refcocog" or task == "ReasonSeg":
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": base64_image,
                    },
                    {"type": "text", "text": "Identify the target referred to by '{}' in the image and return its bounding box in JSON format.".format(sentence)},
                ],
            }
        ]
    elif task == "gres":
        # print("GRES inference")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": base64_image,
                    },
                    {"type": "text", "text": "Identify all instances of the object referred to by '{}' in the image and return a list of their bounding boxes in JSON format. If no object is found, return an empty list [-1, -1, -1, -1].".format(sentence)},
                ],
            }
        ]
    else:
        raise NotImplementedError("Task not supported: {}".format(task))

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    outputs = model.generate(**inputs, max_new_tokens=128)
    generated_ids = outputs
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text 


def get_bbox_mask_urls(refer, idx):
    '''
   从refcoco数据集中提取给定引用索引的边界框、掩码、图像URL和句子信息。

   此函数基于refcoco数据集的Refer类实例，根据索引获取特定引用的相关数据，
   包括对象的边界框坐标、二值掩码、图像文件路径以及关联的句子描述。

   Args:
       refer (object): Refer类的实例，包含refcoco数据集的引用表达式数据。
       idx (int): 要提取数据的引用索引（整数形式）。

   Returns:
       tuple: 包含以下四个元素的元组：
           - bboxs (list or array): 边界框坐标，通常格式为[x, y, width, height]的列表或数组，
             表示对象在图像中的位置和大小。
           - mask_ (array): 二值掩码数组，形状与图像相关，其中True/1表示对象区域，
             False/0表示背景，用于分割任务。
           - image_urls (str): 图像文件的路径或URL字符串，指向引用对应的图像。
           - sent_dict (list of dict): 句子信息字典列表，每个字典包含以下键：
               - 'idx' (int): 句子的顺序索引（从0开始）。
               - 'sent_id' (int): 句子的唯一标识符。
               - 'sent' (str): 句子文本内容（已去除首尾空格）。

   Example:
       示例用法：
       >>> refer = REFER("../DETRIS-main/datasets", task, "unc")
       >>> bboxs, mask_, image_urls, sent_dict = get_bbox_mask_urls(refer, i)
       print(f"图像路径: {image_urls}, 句子数: {len(sent_dict)}")
    '''
    refs = refer.Refs[idx]
    bboxs = refer.getRefBox(idx)
    sentences = refs['sentences']
    image_urls = refer.loadImgs(image_ids=refs['image_id'])[0]
    # cat = cat_process(refs['category_id'])
    image_urls = image_urls['file_name']
    mask = refer.getMask(refs)
    mask_ = mask['mask']
    sent_dict = []
    for i, sent in enumerate(sentences):
        sent_dict.append({
            'idx': i,
            'sent_id': sent['sent_id'],
            'sent': sent['sent'].strip()
        })
    return bboxs, mask_, image_urls, sent_dict


def inference_points_check(
        model, 
        point_candidates, 
        processor, 
        tokenizer, 
        img, 
        sentence, 
        actual_bboxes, 
        mask_, 
        shape_name="star", 
        shape_size =24, 
        use_crop=True, 
        use_coor=False
    ):
    w,h = img.size
    point_outputs_ext = []
    for item in point_candidates:
        x,y = item
        img_ext, color, color_name, adjusted_pos, crop_bbox, size, shape = draw_marker_on_image(img, (x,y), shape=shape_name, size=shape_size, bbox=actual_bboxes)
        if use_coor:
            img_ext = img
        if use_crop:
            img_ext = img_ext.crop(crop_bbox)
        base64_image = image_to_base64(img_ext)
        if use_coor:
            if use_crop:
                x_new = x-crop_bbox[0]
                y_new = y-crop_bbox[1]
                w_new = crop_bbox[2]-crop_bbox[0]
                h_new = crop_bbox[3]-crop_bbox[1]
            else:
                x_new = x
                y_new = y
                w_new = w
                h_new = h

            x_new = round(x_new / w_new * 1000)
            y_new = round(y_new / h_new * 1000)
            x_new = max(0, min(x_new, 1000))
            y_new = max(0, min(y_new, 1000))

            check_messages = get_points_check_messages_corr(base64_image, (x_new,y_new), sentence)
        else:
            check_messages = get_points_check_messages(base64_image, color_name, sentence)
        inputs = processor.apply_chat_template(
            check_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128, output_scores=True, return_dict_in_generate=True,do_sample=False)
        generated_ids = outputs.sequences
        logits = outputs.scores  # 这是包含每一步logits的元组
        analysis_results = analyze_generation_logits(outputs, inputs, tokenizer)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids  , generated_ids)
        ]
        output_text_point = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        point_outputs_ext.append({
            'point': (x,y),
            'color': color_name,
            'model_output': output_text_point[0],
            'label': True if mask_[min(h-1,int(y)),min(w-1,int(x))]>0 else False,
            'analysis_results': analysis_results[0]
        })
    return point_outputs_ext


def coco_xywh_to_xyxy(bbox_xywh):
    """
    将COCO格式的边界框(xywh)转换为xyxy格式。
    
    COCO格式: [x_min, y_min, width, height] - 左上角坐标和宽高
    
    参数:
        bbox_xywh (list/tuple): COCO格式边界框，顺序为[x_min, y_min, width, height]
        
    返回:
        list: xyxy格式边界框，顺序为[x_min, y_min, x_max, y_max]
    """
    x_min, y_min, w, h = bbox_xywh
    x_max = x_min + w
    y_max = y_min + h
    return [x_min, y_min, x_max, y_max]


def random_expand_bbox_all_directions(
    bbox: Union[Tuple[int, int, int, int], List[int]],
    w: int,
    h: int,
    alpha1: float,
    alpha2: float,
    random_seed: int = None
) -> Tuple[int, int, int, int]:
    """
    四个方向（左、右、上、下）均随机扩大bbox（xyxy格式），每个方向扩大比例在[alpha1, alpha2]之间
    确保新bbox不超出图像边界[w, h]
    
    参数:
        bbox: tuple/list - 原始bbox（x1, y1, x2, y2）
        w: int - 图像宽度（x坐标最大值 ≤ w）
        h: int - 图像高度（y坐标最大值 ≤ h）
        alpha1: float - 最小扩大比例（如0.1=10%）
        alpha2: float - 最大扩大比例（如0.3=30%），需满足 alpha2 ≥ alpha1 > 0
        random_seed: int - 随机种子（可选，保证结果可复现）
    
    返回:
        Tuple[int, int, int, int] - 扩大后的新bbox（xyxy格式，整数坐标）
    """
    # 解析原始bbox坐标
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1  # 原始bbox宽度（水平方向）
    bbox_h = y2 - y1  # 原始bbox高度（垂直方向）
    
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
    
    # --------------------------
    # 每个方向独立随机扩大（左、右、上、下）
    # --------------------------
    # 1. 左方向：x1减小，最大不小于0
    alpha_left = random.uniform(alpha1, alpha2)  # 左方向随机比例
    expand_left = int(bbox_w * alpha_left)
    new_x1 = max(0, x1 - expand_left)
    
    # 2. 右方向：x2增大，最大不超过w
    alpha_right = random.uniform(alpha1, alpha2)  # 右方向随机比例
    expand_right = int(bbox_w * alpha_right)
    new_x2 = min(w, x2 + expand_right)
    
    # 3. 上方向：y1减小，最大不小于0
    alpha_up = random.uniform(alpha1, alpha2)  # 上方向随机比例
    expand_up = int(bbox_h * alpha_up)
    new_y1 = max(0, y1 - expand_up)
    
    # 4. 下方向：y2增大，最大不超过h
    alpha_down = random.uniform(alpha1, alpha2)  # 下方向随机比例
    expand_down = int(bbox_h * alpha_down)
    new_y2 = min(h, y2 + expand_down)
    
    return (new_x1, new_y1, new_x2, new_y2)


def random_expand_bbox(
    bbox: Union[Tuple[int, int, int, int], List[int]],
    w: int,
    h: int,
    alpha: float,
    random_seed: int = None
) -> Tuple[int, int, int, int]:
    """
    随机向一个方向扩大xyxy格式的bbox，扩大比例为alpha，确保新bbox不超出图像边界[w, h]
    
    参数:
        bbox: tuple/list - 原始bbox，格式为(x1, y1, x2, y2)（xyxy）
            x1: 左上角x坐标（水平方向，0 ≤ x1 < x2 ≤ w）
            y1: 左上角y坐标（垂直方向，0 ≤ y1 < y2 ≤ h）
            x2: 右下角x坐标
            y2: 右下角y坐标
        w: int - 图像宽度（x坐标最大值，x ≤ w）
        h: int - 图像高度（y坐标最大值，y ≤ h）
        alpha: float - 扩大系数（正浮点数，例如0.2表示扩大20%，1.0表示扩大1倍）
        random_seed: int - 随机种子（可选），用于固定随机方向，保证结果可重复
    
    返回:
        Tuple[int, int, int, int] - 扩大后的新bbox（xyxy格式，整数坐标）
    
    异常:
        TypeError - 输入类型不正确（如bbox不是tuple/list、w/h不是整数）
        ValueError - 输入参数无效（如bbox格式错误、w/h≤0、alpha≤0、bbox超出图像边界）
    """
    # --------------------------
    # 1. 输入验证
    # --------------------------
    # 验证bbox类型和长度
    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        raise TypeError(f"bbox必须是长度为4的tuple或list，当前输入：{bbox}")
    
    # 解析并转换为整数坐标（bbox坐标必须是整数）
    x1, y1, x2, y2 = map(int, bbox)
    
    # 验证图像宽高有效性
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        raise ValueError(f"图像宽高w/h必须是正整数，当前输入：w={w}, h={h}")
    
    # 验证alpha有效性
    if not (isinstance(alpha, (int, float)) and alpha > 0):
        raise ValueError(f"扩大系数alpha必须是正数值，当前输入：{alpha}")
    
    # 验证原始bbox有效性（在图像范围内且x1<x2、y1<y2）
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"原始bbox格式错误，必须满足x1 < x2且y1 < y2，当前输入：({x1}, {y1}, {x2}, {y2})")
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError(
            f"原始bbox超出图像边界，图像范围[w={w}, h={h}]，当前bbox：({x1}, {y1}, {x2}, {y2})"
        )
    
    # --------------------------
    # 2. 计算原始bbox属性和可扩大方向
    # --------------------------
    bbox_w = x2 - x1  # 原始bbox宽度（水平方向长度）
    bbox_h = y2 - y1  # 原始bbox高度（垂直方向长度）
    
    # 定义4个可能的扩大方向：左（x1减小）、右（x2增大）、上（y1减小）、下（y2增大）
    # 每个方向存储：(方向名称, 最大可扩大距离, 扩大后的x1/y1/x2/y2计算函数)
    directions = []
    
    # 左方向：x1最多减小到0，最大可扩距离 = min(bbox_w * alpha, x1)
    max_left = min(int(bbox_w * alpha), x1)
    if max_left > 0:
        directions.append(("左", max_left, lambda d: (x1 - d, y1, x2, y2)))
    
    # 右方向：x2最多增大到w，最大可扩距离 = min(bbox_w * alpha, w - x2)
    max_right = min(int(bbox_w * alpha), w - x2)
    if max_right > 0:
        directions.append(("右", max_right, lambda d: (x1, y1, x2 + d, y2)))
    
    # 上方向：y1最多减小到0，最大可扩距离 = min(bbox_h * alpha, y1)
    max_up = min(int(bbox_h * alpha), y1)
    if max_up > 0:
        directions.append(("上", max_up, lambda d: (x1, y1 - d, x2, y2)))
    
    # 下方向：y2最多增大到h，最大可扩距离 = min(bbox_h * alpha, h - y2)
    max_down = min(int(bbox_h * alpha), h - y2)
    if max_down > 0:
        directions.append(("下", max_down, lambda d: (x1, y1, x2, y2 + d)))
    
    # 检查是否有可扩大的方向（避免bbox已占满整个图像）
    if not directions:
        print(f"警告：无有效扩大方向（bbox已占满图像或alpha过小），返回原始bbox")
        return (x1, y1, x2, y2)
    
    # --------------------------
    # 3. 随机选择方向并计算新bbox
    # --------------------------
    # 设置随机种子（可选）
    if random_seed is not None:
        random.seed(random_seed)
    
    # 随机选择一个有效方向
    selected_dir, max_expand, expand_func = random.choice(directions)
    
    # 计算实际扩大距离（在0到max_expand之间取整，确保不超出边界）
    # 注：如果需要固定扩大alpha比例（而非最大可扩距离），可直接用int(bbox_w*alpha)或int(bbox_h*alpha)
    actual_expand = random.randint(1, max_expand)  # 至少扩大1个像素（避免无变化）
    
    # 计算新bbox坐标
    new_x1, new_y1, new_x2, new_y2 = expand_func(actual_expand)
    
    # 最终边界校验（双重保障，避免极端情况）
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(w, new_x2)
    new_y2 = min(h, new_y2)
    
    # print(f"扩大完成：方向={selected_dir}，原始bbox={bbox}，新bbox=({new_x1}, {new_y1}, {new_x2}, {new_y2})，扩大距离={actual_expand}")
    
    return (new_x1, new_y1, new_x2, new_y2)

def get_random_point_labels(mask: np.ndarray, bbox: tuple or list, N: int, random_seed: int = None) -> tuple[list, np.ndarray]:
    """
    在bbox范围内从mask数组中随机选取N个点，返回选中点的坐标列表和对应类别标签
    
    参数:
        mask: np.ndarray - 输入的2D mask数组（[height, width]）
        bbox: tuple/list - xyxy格式的边界框，格式为(x1, y1, x2, y2)，其中：
            x1: 左上角x坐标（水平方向，对应mask的列索引）
            y1: 左上角y坐标（垂直方向，对应mask的行索引）
            x2: 右下角x坐标（x2 > x1）
            y2: 右下角y坐标（y2 > y1）
        N: int - 要随机选取的点的数量
        random_seed: int - 随机种子（可选），用于保证结果可重复
    
    返回:
        tuple[list, np.ndarray]:
            第一个元素：选中点的坐标列表，格式为[[x1,y1], [x2,y2], ..., [xn,yn]]
            第二个元素：长度为N的类别标签数组，元素为0（负点）或1（正点）
    
    异常:
        TypeError - 当mask不是numpy数组或bbox格式不正确时抛出
        ValueError - 当N不合法、bbox超出mask范围、bbox格式错误或范围内点数不足时抛出
    """
    # --------------------------
    # 1. 输入类型验证
    # --------------------------
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask必须是numpy数组类型")
    if mask.ndim != 2:
        raise ValueError("mask必须是2D数组（[height, width]），当前维度为{}".format(mask.ndim))
    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        raise TypeError("bbox必须是长度为4的tuple或list，格式为(x1, y1, x2, y2)")
    
    # 解析bbox并转换为整数（坐标必须是整数索引）
    x1, y1, x2, y2 = map(int, bbox)
    mask_h, mask_w = mask.shape  # mask的高度（行）和宽度（列）
    
    # --------------------------
    # 2. Bbox边界有效性验证与裁剪
    # --------------------------
    # 检查bbox坐标逻辑（x2 > x1，y2 > y1）
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox格式错误：必须满足x1 < x2且y1 < y2，当前为({}, {}, {}, {})".format(x1, y1, x2, y2))
    
    # 裁剪bbox到mask的有效范围（避免索引越界）
    x1_clipped = max(0, x1)
    y1_clipped = max(0, y1)
    x2_clipped = min(mask_w, x2)
    y2_clipped = min(mask_h, y2)
    
    # 检查裁剪后的bbox是否有效（避免裁剪后无有效区域）
    if x1_clipped >= x2_clipped or y1_clipped >= y2_clipped:
        raise ValueError(
            "bbox超出mask范围或无效：\n"
            f"mask尺寸：(height={mask_h}, width={mask_w})\n"
            f"输入bbox：({x1}, {y1}, {x2}, {y2})\n"
            f"裁剪后bbox：({x1_clipped}, {y1_clipped}, {x2_clipped}, {y2_clipped})"
        )
    
    # --------------------------
    # 3. N的有效性验证
    # --------------------------
    # 计算bbox范围内的总点数
    bbox_h = y2_clipped - y1_clipped  # bbox的高度（行数）
    bbox_w = x2_clipped - x1_clipped  # bbox的宽度（列数）
    total_bbox_points = bbox_h * bbox_w
    
    if N <= 0:
        raise ValueError(f"N必须是正整数，当前输入为{N}")
    if N > total_bbox_points:
        raise ValueError(
            f"bbox范围内的总点数为{total_bbox_points}，不足以选取{N}个点\n"
            f"bbox范围：高度={bbox_h}，宽度={bbox_w}"
        )
    
    # --------------------------
    # 4. 提取bbox范围内的mask子区域
    # --------------------------
    mask_bbox = mask[y1_clipped:y2_clipped, x1_clipped:x2_clipped]  # [y切片, x切片]对应[行, 列]
    
    # --------------------------
    # 5. 随机选取点并计算坐标和类别
    # --------------------------
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 生成bbox子区域内的扁平化随机索引（无放回抽样）
    random_indices = np.random.choice(total_bbox_points, size=N, replace=False)
    
    # 将扁平化索引转换为子区域内的（行索引，列索引）
    # 注意：np.unravel_index返回的是（row, col），对应子区域的y、x坐标
    bbox_row_indices, bbox_col_indices = np.unravel_index(random_indices, shape=mask_bbox.shape)
    
    # 转换为全局mask的坐标（x=列索引，y=行索引）
    global_x = bbox_col_indices + x1_clipped
    global_y = bbox_row_indices + y1_clipped
    
    # 整理坐标为[[x1,y1], [x2,y2], ...]格式的列表
    coordinates = np.stack([global_x, global_y], axis=1).tolist()
    
    # 计算类别标签
    selected_values = mask_bbox.flatten()[random_indices]
    labels = (selected_values > 0).astype(int)
    
    return coordinates, labels

