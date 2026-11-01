"""知覚: オープン語彙物体検出 (Grounding DINO) + 深度画像からの 3D 位置・姿勢推定。

AI には RGB-D カメラ画像だけを与え、シミュレータ内部の物体座標は一切使わない。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
DETECTOR_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"  # 動作確認したコミット

# ロボットが「知っている物体」のカタログ (シーンに無い物も含む)。
# Grounding DINO は候補語を並べて競わせると色・形の識別が安定するため、毎回まとめて問い合わせる。
# 色違い・形違いの紛らわしい候補 (green cylinder 等) を入れると誤ラベルが出たため、
# シーンに無い候補は意味的に離れた日用品にしている (6 シーン評価: 42/42 正解)。
CATALOG = [
    "red cube", "green cube", "blue cube", "yellow ball", "orange cylinder",
    "wooden box", "white plate",
    "banana", "coffee mug", "smartphone",
]
CONTAINERS = ["wooden box", "white plate"]
DISTRACTORS = ["robot arm"]  # ロボット自身を物体として誤検出しないための吸収用ラベル


@dataclass
class Detection:
    label: str
    score: float
    box: np.ndarray  # [x0, y0, x1, y1] pixel
    # 3D 推定結果 (world 座標)
    center: np.ndarray | None = None  # 上面中心 (x, y) と上面高さ z
    top_z: float = 0.0
    yaw: float = 0.0
    extent: np.ndarray | None = None  # xy の min/max (容器の範囲判定用)
    surface_z: float = 0.0  # 容器の底面/皿の上面高さ
    n_points: int = 0

    @property
    def is_container(self) -> bool:
        return self.label in CONTAINERS


@dataclass
class SceneState:
    detections: list[Detection]
    rgb: np.ndarray
    depth: np.ndarray
    seconds: float = 0.0
    raw: list = field(default_factory=list)  # NMS 前の候補 (可視化用)

    def get(self, label: str) -> Detection | None:
        c = [d for d in self.detections if d.label == label]
        return max(c, key=lambda d: d.score) if c else None

    @property
    def objects(self):
        return [d for d in self.detections if not d.is_container]

    @property
    def containers(self):
        return [d for d in self.detections if d.is_container]


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0



class Detector:
    """Grounding DINO によるオープン語彙物体検出 (モデルは 1 回だけ読み込んで共有する)。"""

    def __init__(self, model_id: str = DETECTOR_ID):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        rev = DETECTOR_REVISION if model_id == DETECTOR_ID else None
        self.proc = AutoProcessor.from_pretrained(model_id, revision=rev)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=rev).eval()

    def detect(self, rgb: np.ndarray, phrases: list[str], box_thr: float = 0.30):
        """各候補語のトークン区間ごとにスコアを集計し、各ボックスに最尤ラベルを割り当てる。"""
        from PIL import Image

        torch = self.torch
        h, w = rgb.shape[:2]
        text = ". ".join(phrases) + "."
        enc = self.proc.tokenizer(text, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        spans, pos = [], 0
        for ph in phrases:
            s = text.index(ph, pos)
            e = s + len(ph)
            pos = e
            spans.append([i for i, (a, b) in enumerate(offsets) if b > a and a >= s and b <= e])
        image = Image.fromarray(rgb)
        with torch.no_grad():
            inputs = self.proc(images=image, text=text, return_tensors="pt")
            out = self.detector(**inputs)
        logits = out.logits.sigmoid()[0].numpy()  # (Q, T)
        boxes = out.pred_boxes[0].numpy()  # (Q, 4) cx, cy, w, h (正規化)
        scores = np.stack([logits[:, sp].mean(1) for sp in spans], 1)  # (Q, P)
        best = scores.max(1)
        cand = []
        for q in np.argsort(-best):
            if best[q] < box_thr:
                break
            cx, cy, bw, bh = boxes[q]
            b = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
            cand.append(Detection(phrases[int(scores[q].argmax())], float(best[q]), b))
        # クラス非依存 NMS (同じ物体に複数ラベルが付くのを防ぐ)
        keep = []
        for d in cand:
            if all(_iou(d.box, k.box) < 0.5 for k in keep):
                keep.append(d)
        return keep, cand



class Perception:
    def __init__(self, model: mujoco.MjModel, renderer: mujoco.Renderer, detector: "Detector | None",
                 camera: str = "perception"):
        # レンダラ (OpenGL コンテキスト) は 1 つを使い回す。複数作ると深度画像が壊れるため。
        self.m = model
        self.r = renderer
        self.det = detector
        self.cam = model.camera(camera).id
        self.w, self.h = renderer.width, renderer.height
        fovy = np.deg2rad(model.cam_fovy[self.cam])
        self.f = (self.h / 2) / np.tan(fovy / 2)

    # ------------------------------------------------------------------ センサ
    def capture(self, data: mujoco.MjData):
        self.r.disable_depth_rendering()
        self.r.update_scene(data, camera=self.cam)
        rgb = self.r.render().copy()
        self.r.enable_depth_rendering()
        self.r.update_scene(data, camera=self.cam)
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        self._cam_pos = data.cam_xpos[self.cam].copy()
        self._cam_mat = data.cam_xmat[self.cam].reshape(3, 3).copy()
        return rgb, depth

    def backproject(self, us, vs, depth_vals):
        x = (us - self.w / 2 + 0.5) / self.f * depth_vals
        y = -(vs - self.h / 2 + 0.5) / self.f * depth_vals
        pc = np.stack([x, y, -depth_vals], axis=-1)
        return pc @ self._cam_mat.T + self._cam_pos

    def project(self, p_world):
        pc = (np.asarray(p_world) - self._cam_pos) @ self._cam_mat
        u = pc[..., 0] / -pc[..., 2] * self.f + self.w / 2 - 0.5
        v = -pc[..., 1] / -pc[..., 2] * self.f + self.h / 2 - 0.5
        return np.stack([u, v], -1)

    # ------------------------------------------------------------------ 検出
    def detect(self, rgb: np.ndarray, phrases: list[str], box_thr: float = 0.30):
        return self.det.detect(rgb, phrases, box_thr)
    # ------------------------------------------------------------------ 3D 推定
    def localize(self, det: Detection, depth: np.ndarray) -> Detection:
        x0, y0, x1, y1 = det.box
        mx, my = (x1 - x0) * 0.12, (y1 - y0) * 0.12
        u0, u1 = int(max(0, x0 + mx)), int(min(self.w, x1 - mx))
        v0, v1 = int(max(0, y0 + my)), int(min(self.h, y1 - my))
        if u1 <= u0 or v1 <= v0:
            return det
        vs, us = np.mgrid[v0:v1, u0:u1]
        dv = depth[v0:v1, u0:u1]
        pts = self.backproject(us.ravel().astype(float), vs.ravel().astype(float), dv.ravel())
        if det.is_container:
            pts = pts[(pts[:, 2] > 0.002) & (pts[:, 2] < 0.15)]
            if len(pts) < 20:
                return det
            det.extent = np.array([np.percentile(pts[:, 0], 3), np.percentile(pts[:, 0], 97),
                                   np.percentile(pts[:, 1], 3), np.percentile(pts[:, 1], 97)])
            cx, cy = (det.extent[0] + det.extent[1]) / 2, (det.extent[2] + det.extent[3]) / 2
            det.top_z = float(np.percentile(pts[:, 2], 97))
            hx, hy = (det.extent[1] - det.extent[0]) / 2 - 0.02, (det.extent[3] - det.extent[2]) / 2 - 0.02
            inner = pts[(np.abs(pts[:, 0] - cx) < hx) & (np.abs(pts[:, 1] - cy) < hy)]
            # 底面/皿の上面: 既に置かれた物体の上面に引きずられないよう低い分位点を使う
            det.surface_z = float(np.percentile(inner[:, 2], 10)) if len(inner) > 10 else det.top_z
            det.center = np.array([cx, cy, det.top_z])
            det.n_points = len(pts)
            return det
        # 手前の面だけを残す (背後のテーブルや別物体を除外)
        d_near = np.percentile(dv, 3)
        sel = (dv.ravel() < d_near + 0.07)
        pts = pts[sel]
        pts = pts[(pts[:, 2] > 0.004) & (pts[:, 2] < 0.30)]
        if len(pts) < 10:
            return det
        top = float(np.percentile(pts[:, 2], 95))
        top_pts = pts[pts[:, 2] > top - 0.005]  # 側面の点が混ざらないよう上面だけ
        det.top_z = top
        det.center = np.array([np.median(top_pts[:, 0]), np.median(top_pts[:, 1]), top])
        det.yaw = _estimate_yaw(top_pts[:, :2])
        det.n_points = len(pts)
        return det

    def observe(self, data: mujoco.MjData, phrases=None) -> SceneState:
        t0 = time.time()
        phrases = phrases or (CATALOG + DISTRACTORS)
        rgb, depth = self.capture(data)
        dets, raw = self.detect(rgb, phrases)
        dets = [d for d in dets if d.label not in DISTRACTORS]
        # 同じラベルは最高スコアの 1 つだけ採用 (本シーンは各物体が 1 個ずつ)
        uniq = {}
        for d in dets:
            if d.label not in uniq or d.score > uniq[d.label].score:
                uniq[d.label] = d
        dets = [self.localize(d, depth) for d in uniq.values()]
        dets = [d for d in dets if d.center is not None]
        return SceneState(dets, rgb, depth, time.time() - t0, raw)


def _estimate_yaw(xy: np.ndarray) -> float:
    """上面の点群を最小面積で囲む矩形の向き (0〜90°) を総当たりで求める。"""
    if len(xy) < 15:
        return 0.0
    xy = xy - xy.mean(0)
    best, best_area = 0.0, np.inf
    for a in np.deg2rad(np.arange(0, 90, 2)):
        c, s = np.cos(a), np.sin(a)
        r = xy @ np.array([[c, -s], [s, c]])
        area = np.ptp(r[:, 0]) * np.ptp(r[:, 1])
        if area < best_area:
            best, best_area = a, area
    return float(best)


def describe_position(p: np.ndarray) -> str:
    """ユーザー (認識カメラ側) から見た大まかな位置表現。y+ が右、x+ が手前。"""
    lr = "right" if p[1] > 0.07 else "left" if p[1] < -0.07 else "center"
    fb = "front" if p[0] > 0.56 else "back" if p[0] < 0.44 else ""
    return f"{fb} {lr}".strip()
