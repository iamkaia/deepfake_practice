# explain.py
import json
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam
from pathlib import Path
import configs
from model import build_lora_clip, build_classifier
from dataset import make_loaders
from transformers import CLIPProcessor
from PIL import Image


def visualize_errors():
    data=json.load(open(configs.OUTPUT_DIR/"experiment_log.json"))
    recs=data['records']
    mis=[]
    for r in recs:
        for p,pr in zip(r['frame_paths'],r['frame_preds']):
            if r['true_label']==1 and pr==0:
                mis.append(p)
                if len(mis)>=3: break
        if len(mis)>=3: break
    '''
    # 分類 taxonomy
    taxonomy={'illumination':[mis[0]],'occlusion':[mis[1]],'pose':[mis[2]]}
    print("Error Taxonomy:", taxonomy)
    # 可視化
    '''
    clip=build_lora_clip().to(configs.DEVICE)
    clf=build_classifier().to(configs.DEVICE)
    ckpt=torch.load(configs.CHECKPOINT_DIR/"lora_clip.pth",map_location=configs.DEVICE)
    clip.vision_model.load_state_dict(ckpt['vision_state'])
    clf.load_state_dict(ckpt['clf_state'])
    clip.eval()
    clf.eval()
    proc=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",use_fast=True)
    cam=LayerGradCam(lambda x: clf(clip.get_image_features(pixel_values=x)),clip.vision_model.embeddings.patch_embedding)
    for path in mis:
        img=Image.open(path).convert('RGB')
        plt.figure()
        plt.imshow(img)
        plt.title('原始')
        plt.axis('off')
        
        t=proc(images=img,return_tensors='pt')['pixel_values'].to(configs.DEVICE)
        attr=cam.attribute(t,target=1)
        heat = F.interpolate(attr.mean(1,keepdim=True), size=(224,224), mode='bilinear', align_corners=False)
        heat = heat.squeeze().detach().cpu().numpy()
        plt.figure()
        plt.imshow(img)
        plt.imshow(heat,alpha=0.5)
        plt.title('Grad-CAM')
        plt.axis('off')
    print("Discussion: 參考R1–R4例子，改善建議：prompt-tuning、side-decoderAdapter等。")

