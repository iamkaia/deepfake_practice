# evaluate.py
import json
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import configs
from model import build_lora_clip, build_classifier
from dataset import make_loaders
from transformers import CLIPModel, CLIPProcessor, logging

def evaluate_and_log():
    # 載入模型與分類頭
    clip = build_lora_clip().to(configs.DEVICE)
    clf  = build_classifier().to(configs.DEVICE)
    ckpt= torch.load(configs.CHECKPOINT_DIR/"lora_clip.pth", map_location=configs.DEVICE)
    clip.vision_model.load_state_dict(ckpt['vision_state'])
    clf.load_state_dict(       ckpt['clf_state']   )
    clip.eval(); clf.eval()

    # 取得測試 loader 與樣本列表
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    _, _, test_loader, test_samples = make_loaders(proc)

    all_p, all_y, all_vid = [], [], []
    for imgs, labels in test_loader:
        imgs = imgs.to(configs.DEVICE)
        with torch.no_grad(): feats = clip.get_image_features(pixel_values=imgs)
        probs = torch.softmax(clf(feats), dim=1)[:,1].cpu().numpy()
        all_p.extend(probs.tolist()); all_y.extend(labels.tolist())
        # vid 需要從 DataLoader 額外返回路徑
    # Frame-level
    auc = roc_auc_score(all_y, all_p)
    acc = accuracy_score(all_y, [p>=configs.THRESHOLD for p in all_p])
    f1  = f1_score(all_y, [p>=configs.THRESHOLD for p in all_p])

    # Video-level 聚合
    # ...
    records = []
    overall = {"frame_auc":auc, "frame_acc":acc, "frame_f1":f1}
    with open(configs.OUTPUT_DIR/"experiment_log.json","w") as f:
        json.dump({"records":records, "overall":overall}, f, indent=2)
