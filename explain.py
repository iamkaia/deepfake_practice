# explain.py
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam
import configs
from model import build_lora_clip, build_classifier
from dataset import make_loaders

def visualize_errors():
    # 載入和 evaluate.py 一致
    clip = build_lora_clip().to(configs.DEVICE)
    clf  = build_classifier().to(configs.DEVICE)
    ckpt= torch.load(configs.CHECKPOINT_DIR/"lora_clip.pth", map_location=configs.DEVICE)
    clip.vision_model.load_state_dict(ckpt['vision_state'])
    clf.load_state_dict(       ckpt['clf_state']   )
    clip.eval(); clf.eval()

    # 取得前三個誤判樣本
    # ... 呼叫 evaluate.py 內蒐集結果
    mis = []
    cam = LayerGradCam(lambda x: clf(clip.get_image_features(pixel_values=x)),
                       clip.vision_model.embeddings.patch_embedding)
    for path, p, y in mis:
        img = dataset_preprocess(path).unsqueeze(0).to(configs.DEVICE)
        attr = cam.attribute(img, target=1)
        heat = F.interpolate(attr.mean(1,keepdim=True),(224,224),'bilinear').squeeze().cpu().numpy()
        orig = recover_image(img)
        plt.subplot(1,2,1); plt.imshow(orig); plt.title('原圖'); plt.axis('off')
        plt.subplot(1,2,2); plt.imshow(orig); plt.imshow(heat,alpha=0.5); plt.title('Grad-CAM'); plt.axis('off')
        plt.show()
    print("分析討論：...")