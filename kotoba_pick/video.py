"""動画合成: シミュレーション映像 + AI の内部状態 (認識結果・LLM 計画) を 1 画面に重ねて MP4 に書き出す。"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable, Optional

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 30
MAIN_BOX = (16, 64, 856, 536)  # x0, y0, x1, y1 (840x472)
PERC_BOX = (872, 64, 1264, 358)  # 392x294
PLAN_BOX = (872, 370, 1264, 704)
TEXT_BOX = (16, 548, 856, 704)

BG = (17, 20, 26)
PANEL = (28, 33, 42)
LINE = (55, 63, 78)
FG = (232, 236, 242)
SUB = (150, 160, 176)
ACCENT = (86, 182, 255)
OK = (96, 214, 140)
WARN = (255, 186, 73)
NG = (255, 99, 99)

JA_LABEL = {
    "red cube": "赤いブロック", "green cube": "緑のブロック", "blue cube": "青いブロック",
    "yellow ball": "黄色いボール", "orange cylinder": "オレンジの缶", "wooden box": "木箱", "white plate": "白い皿",
    "banana": "バナナ", "coffee mug": "マグカップ", "smartphone": "スマホ",
}
BOX_COLOR = {
    "red cube": (255, 90, 90), "green cube": (90, 230, 120), "blue cube": (110, 160, 255),
    "yellow ball": (255, 225, 80), "orange cylinder": (255, 160, 60), "wooden box": (210, 160, 110),
    "white plate": (235, 235, 235),
}


def ja(label: str) -> str:
    return JA_LABEL.get(label, label)


_FONT_CACHE: dict = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        names = (["BIZ-UDGothicB.ttc", "YuGothB.ttc", "meiryob.ttc"] if bold
                 else ["BIZ-UDGothicR.ttc", "YuGothM.ttc", "meiryo.ttc", "msgothic.ttc"])
        for n in names:
            p = pathlib.Path("C:/Windows/Fonts") / n
            if p.exists():
                _FONT_CACHE[key] = ImageFont.truetype(str(p), size)
                break
        else:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


@dataclasses.dataclass
class UIState:
    title: str = "KotobaPick — 言葉で動くお片付けロボット"
    scenario: str = ""
    instruction: str = ""
    instr_reveal: float = 1.0
    reply: str = ""
    plan: list = dataclasses.field(default_factory=list)  # [(pick, place)]
    step_status: list = dataclasses.field(default_factory=list)  # pending/active/done/failed/retry
    plan_caption: str = ""
    status: str = ""
    caption: str = ""
    caption_color: tuple = ACCENT
    perc_img: Optional[np.ndarray] = None
    perc_dets: list = dataclasses.field(default_factory=list)
    perc_reveal: float = 1.0
    perc_caption: str = "ロボットの目 (Grounding DINO)"
    highlight: Optional[str] = None  # 認識パネルで強調するラベル
    markers: list = dataclasses.field(default_factory=list)  # [(pos(3), rgba, radius)]
    speed: float = 1.0
    banner: Optional[str] = None
    banner_sub: str = ""
    banner_alpha: float = 0.0


def _icon(dr, kind, cx, cy):
    r = 8
    if kind == "done":
        dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=OK)
        dr.line([(cx - 4, cy), (cx - 1, cy + 4), (cx + 5, cy - 4)], fill=BG, width=3)
    elif kind == "active":
        dr.polygon([(cx - 5, cy - 7), (cx - 5, cy + 7), (cx + 7, cy)], fill=ACCENT)
    elif kind == "failed":
        dr.line([(cx - 6, cy - 6), (cx + 6, cy + 6)], fill=NG, width=3)
        dr.line([(cx - 6, cy + 6), (cx + 6, cy - 6)], fill=NG, width=3)
    elif kind == "retry":
        dr.arc((cx - r, cy - r, cx + r, cy + r), 40, 340, fill=WARN, width=3)
        dr.polygon([(cx + 4, cy - 9), (cx + 10, cy - 3), (cx + 2, cy - 2)], fill=WARN)
    else:
        dr.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=SUB)


def _crop_rect(dets, w, h, aspect, margin=45):
    """検出結果を囲む領域 (パネルの縦横比に合わせる)。"""
    if not dets:
        return 0, 0, w, h
    bx = np.array([d.box for d in dets])
    x0, y0 = bx[:, 0].min() - margin, bx[:, 1].min() - margin
    x1, y1 = bx[:, 2].max() + margin, bx[:, 3].max() + margin
    cw, ch = x1 - x0, y1 - y0
    if cw / ch < aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    cw, ch = min(cw, w), min(ch, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = min(max(0, cx - cw / 2), w - cw)
    y0 = min(max(0, cy - ch / 2), h - ch)
    return x0, y0, x0 + cw, y0 + ch


def _wrap(draw, text, fnt, width):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if draw.textlength(cur + ch, font=fnt) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


class Director:
    """シミュレーション時間に同期して動画フレームを生成し、AI 推論中は映像を止めて解説を挿入する。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, writer, camera: str = "overview"):
        self.m, self.d = model, data
        self.writer = writer
        self.r = mujoco.Renderer(model, MAIN_BOX[3] - MAIN_BOX[1], MAIN_BOX[2] - MAIN_BOX[0])
        self.cam = camera
        self.ui = UIState()
        self.acc = 0.0
        self.frames = 0
        self.perc_proj: Optional[Callable] = None

    def close(self):
        self.r.close()

    # ------------------------------------------------------------------ フレーム生成
    def on_step(self):
        self.acc += self.m.opt.timestep
        period = self.ui.speed / FPS
        if self.acc >= period:
            self.acc -= period
            self.emit()

    def hold(self, seconds: float, anim: Optional[Callable[[float], None]] = None):
        n = max(1, int(round(seconds * FPS)))
        for k in range(n):
            if anim is not None:
                anim((k + 1) / n)
            self.emit()

    def emit(self):
        self.writer.append_data(self.compose())
        self.frames += 1

    def render_main(self) -> np.ndarray:
        self.r.update_scene(self.d, camera=self.cam)
        scn = self.r.scene
        for pos, rgba, rad in self.ui.markers:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]),
                                np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
            scn.ngeom += 1
        return self.r.render()

    def compose(self) -> np.ndarray:
        ui = self.ui
        img = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(img)
        # ヘッダ
        dr.text((18, 14), ui.title, font=font(26, True), fill=FG)
        if ui.scenario:
            tw = dr.textlength(ui.scenario, font=font(20))
            dr.text((W - 18 - tw, 20), ui.scenario, font=font(20), fill=SUB)
        # メイン映像
        main = self.render_main()
        img.paste(Image.fromarray(main), MAIN_BOX[:2])
        if ui.speed != 1.0:
            tag = f"▶▶ {ui.speed:g}倍速"
            dr.rounded_rectangle((MAIN_BOX[2] - 128, MAIN_BOX[1] + 10, MAIN_BOX[2] - 10, MAIN_BOX[1] + 44), 8,
                                 fill=(0, 0, 0))
            dr.text((MAIN_BOX[2] - 118, MAIN_BOX[1] + 15), tag, font=font(20, True), fill=WARN)
        if ui.status:
            dr.rounded_rectangle((MAIN_BOX[0] + 10, MAIN_BOX[3] - 44, MAIN_BOX[0] + 30 +
                                  dr.textlength("動作: " + ui.status, font=font(20)), MAIN_BOX[3] - 10), 8,
                                 fill=(0, 0, 0))
            dr.text((MAIN_BOX[0] + 20, MAIN_BOX[3] - 39), "動作: " + ui.status, font=font(20), fill=FG)
        self._draw_perception(img, dr)
        self._draw_plan(dr)
        self._draw_text(dr)
        if ui.banner and ui.banner_alpha > 0:
            img = self._draw_banner(img)
        return np.asarray(img)

    def _draw_perception(self, img, dr):
        ui = self.ui
        x0, y0, x1, y1 = PERC_BOX
        dr.rectangle(PERC_BOX, fill=PANEL)
        if ui.perc_img is not None:
            pw, ph = x1 - x0, y1 - y0 - 26
            src = Image.fromarray(ui.perc_img)
            cx0, cy0, cx1, cy1 = _crop_rect(ui.perc_dets, src.width, src.height, pw / ph)
            s = pw / (cx1 - cx0)
            img.paste(src.crop((int(cx0), int(cy0), int(cx1), int(cy1))).resize((pw, ph), Image.BILINEAR),
                      (x0, y0 + 26))
            n_show = int(np.ceil(len(ui.perc_dets) * ui.perc_reveal))
            for det in ui.perc_dets[:n_show]:
                b = det.box
                col = BOX_COLOR.get(det.label, ACCENT)
                wdt = 3 if ui.highlight == det.label else 2
                if ui.highlight and ui.highlight != det.label:
                    col = tuple(int(c * 0.45) for c in col)
                rect = (x0 + (b[0] - cx0) * s, y0 + 26 + (b[1] - cy0) * s,
                        x0 + (b[2] - cx0) * s, y0 + 26 + (b[3] - cy0) * s)
                dr.rectangle(rect, outline=col, width=wdt)
                lab = f"{ja(det.label)} {det.score:.2f}"
                tw = dr.textlength(lab, font=font(14))
                tx = min(max(x0 + 2, rect[0]), x1 - tw - 8)
                ty = max(y0 + 28, rect[1] - 19)
                dr.rectangle((tx, ty, tx + tw + 6, ty + 18), fill=(0, 0, 0))
                dr.text((tx + 3, ty + 1), lab, font=font(14), fill=col)
        dr.rectangle((x0, y0, x1, y0 + 26), fill=LINE)
        dr.text((x0 + 8, y0 + 4), ui.perc_caption, font=font(16, True), fill=FG)

    def _draw_plan(self, dr):
        ui = self.ui
        x0, y0, x1, y1 = PLAN_BOX
        dr.rectangle(PLAN_BOX, fill=PANEL)
        dr.rectangle((x0, y0, x1, y0 + 26), fill=LINE)
        dr.text((x0 + 8, y0 + 4), "タスク計画 (ローカルLLM: Qwen2.5)", font=font(16, True), fill=FG)
        y = y0 + 36
        if not ui.plan and ui.plan_caption:  # 計画を立てた結果、実行する手順がない
            dr.text((x0 + 12, y), "（実行する手順はありません）", font=font(18), fill=SUB)
        for i, (pick, place) in enumerate(ui.plan):
            st = ui.step_status[i] if i < len(ui.step_status) else "pending"
            _icon(dr, st, x0 + 20, y + 11)
            line = f"{i + 1}. {ja(pick)} → {ja(place)}"
            dr.text((x0 + 34, y), line, font=font(18, st == "active"), fill=FG if st != "pending" else SUB)
            y += 30
            if y > y1 - 60:
                break
        if ui.plan_caption:
            lines = _wrap(dr, ui.plan_caption, font(14), x1 - x0 - 20)[:5]
            top = max(y + 4, y1 - 8 - 18 * len(lines))
            for k, ln in enumerate(lines):
                dr.text((x0 + 10, top + k * 18), ln, font=font(14), fill=SUB)

    def _draw_text(self, dr):
        ui = self.ui
        x0, y0, x1, y1 = TEXT_BOX
        dr.rectangle(TEXT_BOX, fill=PANEL)
        dr.text((x0 + 12, y0 + 10), "指示", font=font(16, True), fill=SUB)
        shown = ui.instruction[: int(round(len(ui.instruction) * ui.instr_reveal))]
        dr.text((x0 + 60, y0 + 6), f"「{shown}」" if shown else "", font=font(26, True), fill=FG)
        if ui.reply:
            dr.text((x0 + 12, y0 + 52), "ロボット", font=font(16, True), fill=SUB)
            for k, ln in enumerate(_wrap(dr, ui.reply, font(20), x1 - x0 - 110)[:2]):
                dr.text((x0 + 90, y0 + 50 + k * 26), ln, font=font(20), fill=(200, 230, 255))
        if ui.caption:
            dr.text((x0 + 12, y1 - 32), ui.caption, font=font(17), fill=ui.caption_color)

    def _draw_banner(self, img):
        ui = self.ui
        over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(over)
        a = int(255 * ui.banner_alpha)
        od.rectangle((0, 250, W, 470), fill=(10, 12, 18, int(a * 0.88)))
        tw = od.textlength(ui.banner, font=font(44, True))
        od.text(((W - tw) / 2, 300), ui.banner, font=font(44, True), fill=(255, 255, 255, a))
        if ui.banner_sub:
            for k, ln in enumerate(ui.banner_sub.split("\n")):
                sw = od.textlength(ln, font=font(22))
                od.text(((W - sw) / 2, 372 + k * 32), ln, font=font(22), fill=(190, 205, 225, a))
        return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")


def open_writer(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p",
                              macro_block_size=16, ffmpeg_log_level="error")


def title_card(writer, title: str, lines: list[str], seconds: float, footer: str = ""):
    """タイトル/説明用の静止画カード (フェードイン付き)。"""
    base = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(base)
    tw = dr.textlength(title, font=font(48, True))
    dr.text(((W - tw) / 2, 150), title, font=font(48, True), fill=FG)
    dr.line(((W - 520) / 2, 225, (W + 520) / 2, 225), fill=ACCENT, width=3)
    y = 260
    for ln in lines:
        lw = dr.textlength(ln, font=font(26))
        dr.text(((W - lw) / 2, y), ln, font=font(26), fill=(205, 212, 224))
        y += 46
    if footer:
        fw = dr.textlength(footer, font=font(18))
        dr.text(((W - fw) / 2, H - 60), footer, font=font(18), fill=SUB)
    arr = np.asarray(base).astype(np.float32)
    n = int(seconds * FPS)
    for k in range(n):
        a = min(1.0, (k + 1) / 12, (n - k) / 8)
        writer.append_data((arr * a + np.array(BG) * (1 - a)).astype(np.uint8))
