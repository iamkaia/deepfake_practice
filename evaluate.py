# evaluate.py
import json, torch, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
import configs
from model import build_lora_clip, build_classifier
from dataset import make_loaders
from transformers import CLIPProcessor


def evaluate_and_log():
    # LoRA 評估
    clip = build_lora_clip().to(configs.DEVICE)
    clr = build_classifier().to(configs.DEVICE)
    ckpt = torch.load(configs.CHECKPOINT_DIR/"lora_clip.pth", map_location=configs.DEVICE)
    clip.vision_model.load_state_dict(ckpt['vision_state'])
    clr.load_state_dict(ckpt['clf_state'])
    clip.eval()
    clr.eval()

    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    _,_,test_loader,test_samples = make_loaders(proc)
    all_p, all_y = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs=imgs.to(configs.DEVICE)
            feats=clip.get_image_features(pixel_values=imgs)
            probs=torch.softmax(clr(feats),dim=1)[:,1].cpu().numpy()
            all_p.extend(probs.tolist()); all_y.extend(labels.tolist())
    preds=[1 if p>=configs.THRESHOLD else 0 for p in all_p]
    auc=roc_auc_score(all_y, all_p)
    acc=accuracy_score(all_y, preds)
    f1=f1_score(all_y,preds)
    fpr,tpr,_=roc_curve(all_y, all_p)
    fnr=1-tpr
    eer_idx=np.argmin(np.abs(fpr-fnr))
    eer=(fpr[eer_idx]+fnr[eer_idx])/2


    # Baseline 評估
    clf_b = build_classifier().to(configs.DEVICE)
    ckpt_b = torch.load(configs.CHECKPOINT_DIR/"baseline_probe.pth", map_location=configs.DEVICE)
    # 只載入分類頭權重
    clf_b.load_state_dict(ckpt_b['clf_state'])
    all_pb=[]
    with torch.no_grad():
        for imgs,_ in test_loader:
            feats=clip.get_image_features(pixel_values=imgs.to(configs.DEVICE))
            pb=torch.softmax(clf_b(feats),dim=1)[:,1].cpu().numpy()
            all_pb.extend(pb.tolist())
    preds_b=[1 if p>=configs.THRESHOLD else 0 for p in all_pb]
    auc_b=roc_auc_score(all_y, all_pb)
    acc_b=accuracy_score(all_y, preds_b)
    f1_b=f1_score(all_y,preds_b)

    # 繪製 ROC
    plt.figure()
    plt.plot(fpr,tpr,label=f"LoRA AUC={auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend()
    plt.savefig(configs.OUTPUT_DIR/'roc_curve.png')

    # 實驗紀錄
    video_dict={}
    for (path,_),p,pr in zip(test_samples, all_p, preds):
        vid=Path(path).parent.name
        video_dict.setdefault(vid,[]).append((path,p,pr,True))
    records=[]
    for vid, lst in video_dict.items():
        records.append({'video_id':vid,'frame_paths':[e[0] for e in lst],'frame_scores':[e[1] for e in lst],'frame_preds':[e[2] for e in lst],'true_label':lst[0][3]})
    overall={'frame_auc':auc,'frame_acc':acc,'frame_f1':f1,'frame_eer':eer,'baseline_auc':auc_b,'baseline_acc':acc_b,'baseline_f1':f1_b}
    configs.OUTPUT_DIR.mkdir(exist_ok=True)
    with open(configs.OUTPUT_DIR/"experiment_log.json","w") as f:
        json.dump({'records':records,'overall':overall}, f, indent=2)
    print("已生成 experiment_log.json 與 roc_curve.png")
