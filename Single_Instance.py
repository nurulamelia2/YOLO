import torch
import cv2
import torchvision
# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, etc.
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       't4gpu.pt')  # custom trained model

class_label = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "Q",
    24: "Y",
    25: "Z"
}


# def draw_bounding_boxes(pred_tensor, result):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 0.7
#     size_of_tensor = list(pred_tensor.size())
#     rows = size_of_tensor[0]
#     for i in range(0, rows):
#         cv2.rectangle(result, (int(pred_tensor[i,0].item()), int(pred_tensor[i,1].item())),
#         (int(pred_tensor[i,2].item()), int(pred_tensor[i,3].item())), (0, 0, 255), 2)

#         text = class_label[int(pred_tensor[i,5].item())] +" " + str(round(pred_tensor[i,4].item(), 2))

#         image = cv2.putText(result, text, (int(pred_tensor[i,0].item())+5, int(pred_tensor[i,1].item())),
#         font, fontScale, (0, 0, 255), 2)

#     return result

def draw_bounding_boxes(pred_tensor, result, threshold=0.5, iou_threshold=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    size_of_tensor = list(pred_tensor.size())
    rows = size_of_tensor[0]
    for i in range(0, rows):
        if pred_tensor[i, 4].item() >= threshold:
            cv2.rectangle(result, (int(pred_tensor[i, 0].item()), int(pred_tensor[i, 1].item())),
                          (int(pred_tensor[i, 2].item()), int(pred_tensor[i, 3].item())), (0, 0, 255), 2)

            text = class_label[int(pred_tensor[i, 5].item())] + \
                " " + str(round(pred_tensor[i, 4].item(), 2))

            image = cv2.putText(result, text, (int(pred_tensor[i, 0].item()) + 5, int(pred_tensor[i, 1].item())),
                                font, fontScale, (0, 0, 255), 2)

    # Perform Non-Maximum Suppression (NMS) based on IoU threshold
    boxes = pred_tensor[:, :4]  # Get the bounding boxes
    scores = pred_tensor[:, 4]  # Get the confidence scores
    indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    pred_tensor = pred_tensor[indices]

    return result


# Images
# im = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

print(results.xyxy[0])  # im predictions (tensor)
# print("Shape of tensor:", results.xyxy[0].size())
# print(results.xyxy[0][1, 1].item())
res = draw_bounding_boxes(results.xyxy[0], img)

cv2.imshow("", res)
cv2.waitKey(0)
print(results.pandas().xyxy[0])  # im predictions (pandas)
