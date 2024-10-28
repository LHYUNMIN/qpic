def calculate_iou(box1, box2):
# box1, box2: [x_min, y_min, x_max, y_max]
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 두 상자의 겹치는 영역을 계산합니다.
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    # 겹치는 영역의 너비와 높이를 계산합니다.
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)

    # 겹치는 영역의 면적과 각 상자의 면적을 계산합니다.
    intersection_area = intersection_width * intersection_height
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    # IoU를 계산합니다.
    iou = intersection_area / (box1_area + box2_area - intersection_area)

    return iou