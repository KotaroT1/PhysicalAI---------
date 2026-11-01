"""検出器の比較 (OWLv2 vs Grounding DINO): レポート用の図 docs/detector_comparison.png を作る。

    python scripts/compare_detectors.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import kotoba_pick  # noqa: E402,F401
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from kotoba_pick.perception import CATALOG, Detector, _iou  # noqa: E402
from kotoba_pick.scene import build_scene, reset_robot  # noqa: E402
from kotoba_pick.video import BOX_COLOR, font, ja  # noqa: E402

PHRASES = CATALOG + ["robot arm"]
PRESENT = ["red cube", "green cube", "blue cube", "yellow ball", "orange cylinder", "wooden box", "white plate"]


def owlv2_detect(img, thr=0.10):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    mid = "google/owlv2-base-patch16-ensemble"
    rev = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
    proc = Owlv2Processor.from_pretrained(mid, revision=rev)
    model = Owlv2ForObjectDetection.from_pretrained(mid, revision=rev).eval()
    queries = [f"a {p}" for p in PHRASES]
    with torch.no_grad():
        out = model(**proc(text=[queries], images=Image.fromarray(img), return_tensors="pt"))
    s = max(img.shape[:2])
    res = proc.post_process_object_detection(out, threshold=thr, target_sizes=torch.tensor([[s, s]]))[0]
    dets = sorted([(PHRASES[int(l)], float(sc), b.numpy()) for sc, l, b in
                   zip(res["scores"], res["labels"], res["boxes"])], key=lambda x: -x[1])
    keep = []
    for d in dets:
        if all(_iou(d[2], k[2]) < 0.5 for k in keep):
            keep.append(d)
    return keep


def draw(img, dets, title):
    pil = Image.fromarray(img).crop((120, 230, 900, 520)).resize((780 * 2 // 2, 290))
    dr = ImageDraw.Draw(pil)
    for lab, sc, b in dets:
        if lab == "robot arm" or sc < 0.12 or (b[2] - b[0]) * (b[3] - b[1]) > 0.25 * img.shape[0] * img.shape[1]:
            continue
        bb = [b[0] - 120, b[1] - 230, b[2] - 120, b[3] - 230]
        col = BOX_COLOR.get(lab, (255, 80, 255))
        dr.rectangle(bb, outline=col, width=3)
        t = f"{ja(lab)} {sc:.2f}"
        dr.rectangle((bb[0], bb[1] - 18, bb[0] + dr.textlength(t, font=font(14)) + 6, bb[1]), fill=(0, 0, 0))
        dr.text((bb[0] + 3, bb[1] - 17), t, font=font(14), fill=col)
    canvas = Image.new("RGB", (pil.width, pil.height + 40), (20, 22, 28))
    canvas.paste(pil, (0, 40))
    ImageDraw.Draw(canvas).text((10, 8), title, font=font(20, True), fill=(235, 235, 235))
    return canvas


def main():
    model, _ = build_scene(seed=1)
    data = mujoco.MjData(model)
    reset_robot(model, data)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, 720, 960)
    r.update_scene(data, camera="perception")
    img = r.render().copy()
    r.close()

    owl = owlv2_detect(img)
    gd_keep, _ = Detector().detect(img, PHRASES)
    gd = [(d.label, d.score, d.box) for d in gd_keep]

    def summary(dets):
        labs = {}
        for lab, sc, _ in dets:
            labs.setdefault(lab, sc)
        return sum(1 for p in PRESENT if p in labs), [lab for lab in labs if lab not in PRESENT + ["robot arm"]]

    so, sg = summary(owl), summary(gd)
    print("OWLv2:", so, "\nGroundingDINO:", sg)
    def title(name, dets, sm):
        found = {lab for lab, sc, b in dets if sc >= 0.12}
        miss = [ja(p) for p in PRESENT if p not in found]
        return f"{name}: 正しいラベル {sm[0]}/7" + (f"（{'・'.join(miss)} は誤認または未検出）" if miss else "（全物体を正しく判別）")

    a = draw(img, owl, title("OWLv2 base", owl, so))
    b = draw(img, gd, title("Grounding DINO tiny", gd, sg))
    out = Image.new("RGB", (a.width, a.height + b.height + 8), (20, 22, 28))
    out.paste(a, (0, 0))
    out.paste(b, (0, a.height + 8))
    (ROOT / "docs").mkdir(exist_ok=True)
    out.save(ROOT / "docs" / "detector_comparison.png")
    print("saved docs/detector_comparison.png")


if __name__ == "__main__":
    main()
