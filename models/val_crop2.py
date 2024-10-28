import torch
import pickle
from .detection import DETRdemo
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .self import selfAttPatch
from .calculate import calculate_iou
from .box_cxcy import  box_cxcywh_to_xyxy
from .position import  add_1d_positional_embeddings
import torchvision.transforms.functional as F
from torch import nn
from torchvision import models
from torchvision.models import resnet50
from torchvision.transforms import functional as F
import torchvision.transforms as T
import torchvision.transforms as transforms


CLASSES = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
           'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
           'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
           'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
           'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
           'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
           'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
           'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
           'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
           'toothbrush']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
detr = DETRdemo(num_classes=91)

state_dict = torch.hub.load_state_dict_from_url(
url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval()









def process_images_and_extract_features_val(images):
    global final
    
    final = None     
    ss=iter(images)
   
    batch = next(ss)
    image, labels = batch
    resnet_model = resnet50(pretrained=True)

    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model.to(device)
    resnet_model.eval()

    all_images=[]

    for tensor in image.tensors:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ten = tensor.unsqueeze(0).to(device)
        
        all_images.append(ten)
        final_outputs = []

    for out in all_images:
        
        detr.to(device) 
        outputs = detr(out)
     
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.85
        
        img_w, img_h = image.tensors.shape[2:]
        b = box_cxcywh_to_xyxy(outputs['pred_boxes'][0, keep].cpu())
        img_w = torch.tensor(img_w, dtype=torch.float32).to(device)
        img_h = torch.tensor(img_h, dtype=torch.float32).to(device)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        bboxes_scaled = b
        labels = probas.argmax(-1)

        # scores = probas[keep]
   

        human_bboxes = []
        object_bboxes = []

        for p, box, label in zip(probas[keep], bboxes_scaled, labels[keep]):
            if CLASSES[label] == 'person':
                human_bboxes.append((p, box, CLASSES[label]))
            else:
                object_bboxes.append((p, box, CLASSES[label]))
       
        
        max_iou = 0
        best_human_bbox = None
        best_object_bbox = None
 
        for human_bbox in human_bboxes:
            for object_bbox in object_bboxes:
                iou = calculate_iou(human_bbox[1], object_bbox[1])
             
                if iou > max_iou:
                    max_iou = iou
                    best_human_bbox = human_bbox
                    best_object_bbox = object_bbox
       
        # 최대 IoU를 가진 human과 object Bounding Box의 합집합 계산
        if best_human_bbox is not None and best_object_bbox is not None:
            
            human_x1, human_y1, human_x2, human_y2 = best_human_bbox[1]
            object_x1, object_y1, object_x2, object_y2 = best_object_bbox[1]
  
            x1 = min(human_x1, object_x1)
            y1 = min(human_y1, object_y1)
            x2 = max(human_x2, object_x2)
            y2 = max(human_y2, object_y2)
      
        
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
       

            image_cropped_tensor = out[:, :, y1:y2, x1:x2]
            image_cropped_tensor_cpu = image_cropped_tensor[0].cpu()

            image_cropped_np = image_cropped_tensor_cpu.numpy()
            image_cropped_np = np.transpose(image_cropped_np, (1, 2, 0))

            patch_height = image_cropped_tensor.shape[2] // 2
            patch_width = image_cropped_tensor.shape[3] // 2
            pathches = []
            for i in range(2):
                for j in range(2):
                    patch = image_cropped_tensor[:, :, i * patch_height: (i + 1) * patch_height,
                            j * patch_width: (j + 1) * patch_width]
                    pathches.append(patch)

            transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0., std=1.)
                ])

            final_outputs = []
            for patch in pathches:

                
                patch_pil = transforms.ToPILImage()(patch[0].cpu())
                
                # Apply the transformation to the PIL Image
                patch_transformed = transform(patch_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    patch_feature = resnet_model(patch_transformed)
                
                final_outputs.append(patch_feature)    
                
                
                # model = resnet50(pretrained=True)
                # model.eval()
                # model_weights = []
                # conv_layers = []
                # model_children = list(model.children())
                # counter = 0
                # for i in range(len(model_children)):
                #     if type(model_children[i]) == nn.Conv2d:
                #         counter += 1
                #         model_weights.append(model_children[i].weight)
                #         conv_layers.append(model_children[i])
                #     elif type(model_children[i]) == nn.Sequential:
                #         for j in range(len(model_children[i])):
                #             for child in model_children[i][j].children():
                #                 if type(child) == nn.Conv2d:
                #                     counter += 1
                #                     model_weights.append(child.weight)
                #                     conv_layers.append(child)
                

                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # model = model.to(device)
                # patch_pil = transforms.ToPILImage()(patch[0].cpu())
                # image_resnet = transform_resnet(patch_pil)
                # image_resnet = image_resnet.unsqueeze(0).to(device)
                # patch_pil = transforms.ToPILImage()(patch[0].cpu())

                # outputs = []
                # names = []

                # for layer in conv_layers[0:]:
                #     image_resnet = layer(image_resnet)
                    
                
                
                #     outputs.append(image_resnet)
                #     names.append(str(layer))
                    
                # final_outputs.append(outputs[-1])    
                
                
            x0 = final_outputs[0]
            x1 = final_outputs[1] 
            x2 = final_outputs[2] 
            x3 = final_outputs[3]
    
            x0 = add_1d_positional_embeddings(x0)
            x1 = add_1d_positional_embeddings(x1)
            x2 = add_1d_positional_embeddings(x2)
            x3 = add_1d_positional_embeddings(x3)

                
            SA0 = selfAttPatch(x0, x0, x0)
            SA1 = selfAttPatch(x1, x1, x1)
            SA2 = selfAttPatch(x2, x2, x2)
            SA3 = selfAttPatch(x3, x3, x3)

            CA0 = selfAttPatch(x0, x1, x1) + selfAttPatch(x0, x2, x2) + selfAttPatch(x0, x3, x3)
            CA1 = selfAttPatch(x1, x0, x0) + selfAttPatch(x1, x2, x2) + selfAttPatch(x1, x3, x3)
            CA2 = selfAttPatch(x2, x0, x0) + selfAttPatch(x2, x1, x1) + selfAttPatch(x2, x3, x3)
            CA3 = selfAttPatch(x3, x0, x0) + selfAttPatch(x3, x1, x1) + selfAttPatch(x3, x2, x2)

            final0=SA0+CA0
            final1=SA1+CA1
            final2=SA2+CA2
            final3=SA3+CA3
            final = torch.cat([final0, final1, final2, final3], dim=1)
            final = final.view(1, 8200, -1)
            final=final.permute(1,0,2)
            final=final.expand(-1, 4, -1)

            return final
                
        