import torch
import numpy as np


def IOU(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1]), \
                     np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])

    w, h = np.maximum(0, x2 - x1), np.maximum(0, y2 - y1)
    area_stack = w * h
    area_union = box_area + boxes_area - area_stack

    if isMin:
        return area_stack / np.minimum(box_area, boxes_area)
    else:
        return area_stack / area_union


def NMS(boxes, threshold, isMin = False):
    """
    None Maximum Supression
    :param boxes: the boxes for NMS
    :param threshold: IOU threshold
    :return:
    """
    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[np.argsort(-boxes[:, 4])]
    r_box = []
    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]
        r_box.append(a_box)
        index = np.where(IOU(a_box, b_boxes, isMin) < threshold)
        _boxes = b_boxes[index]

    if _boxes.shape[0] == 1:  # 当_boxes中只剩一个元素，则该元素则应加入r_boxes
        r_box.append(_boxes[0])

    return np.stack(r_box)

def Convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox


if __name__ == "__main__":
    a = np.array([10, 10, 50, 50])
    bs = np.array([[10, 10, 50, 50]])
    print(IOU(a, bs))

    # bs = np.array([[1, 1, 10, 10, 0.98], [1, 1, 9, 9, 0.8], [9, 8, 13, 20, 0.7], [6, 11, 18, 17, 0.85]])
    # # print((-bs[:, 4]).argsort())
    # print(NMS(bs, 0.3))
