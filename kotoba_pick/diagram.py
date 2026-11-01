"""システム構成図 (動画のカード / レポートの図として使用)。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .video import ACCENT, BG, FG, FPS, H, OK, PANEL, SUB, W, WARN, font


def _box(dr, xy, title, lines, color):
    x0, y0, x1, y1 = xy
    dr.rounded_rectangle(xy, 12, fill=PANEL, outline=color, width=3)
    tw = dr.textlength(title, font=font(22, True))
    dr.text(((x0 + x1 - tw) / 2, y0 + 12), title, font=font(22, True), fill=color)
    for k, ln in enumerate(lines):
        lw = dr.textlength(ln, font=font(16))
        dr.text(((x0 + x1 - lw) / 2, y0 + 48 + k * 22), ln, font=font(16), fill=FG)


def _arrow(dr, p0, p1, color=SUB, width=3):
    dr.line([p0, p1], fill=color, width=width)
    v = np.array(p1, float) - np.array(p0, float)
    v /= np.linalg.norm(v) + 1e-9
    n = np.array([-v[1], v[0]])
    tip = np.array(p1, float)
    dr.polygon([tuple(tip), tuple(tip - 12 * v + 6 * n), tuple(tip - 12 * v - 6 * n)], fill=color)


def render(llm_name: str = "Qwen2.5-1.5B") -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    title = "システム構成 (すべてローカルPCのCPUで実行)"
    tw = dr.textlength(title, font=font(34, True))
    dr.text(((W - tw) / 2, 34), title, font=font(34, True), fill=FG)

    bw, bh = 250, 128
    row1, row2 = 130, 380
    xs = [60, 370, 680, 990]
    _box(dr, (xs[0], row1, xs[0] + bw, row1 + bh), "日本語の指示", ["「丸いものは全部", "お皿に…」"], FG)
    _box(dr, (xs[1], row1, xs[1] + bw, row1 + bh), "① シーン認識",
         ["Grounding DINO (検出)", "+ 深度画像 → 3D位置", "上面点群 → 向き推定"], ACCENT)
    _box(dr, (xs[2], row1, xs[2] + bw, row1 + bh), "② タスク計画",
         [f"LLM: {llm_name}", "「何を→どこへ」に構造化", "各物体の当てはまりを判定"], ACCENT)
    _box(dr, (xs[3], row1, xs[3] + bw, row1 + bh), "実行順の決定",
         ["積み上げの依存関係を", "解決して手順に並べる"], FG)
    _box(dr, (xs[3], row2, xs[3] + bw, row2 + bh), "③ 再認識",
         ["各手順の直前に", "対象と置き場所を", "カメラで探し直す"], ACCENT)
    _box(dr, (xs[2], row2, xs[2] + bw, row2 + bh), "④⑤ 把持・配置",
         ["減衰最小二乗IK", "最小躍度軌道", "空きスペースを選んで置く"], WARN)
    _box(dr, (xs[1], row2, xs[1] + bw, row2 + bh), "⑥ 結果確認",
         ["位置と高さを画像で検証", "失敗・落下は検知して再試行", "全手順後に最終確認"], OK)
    _box(dr, (xs[0], row2, xs[0] + bw, row2 + bh), "MuJoCo",
         ["Franka Panda 7軸アーム", "RGB-Dカメラ", "物理シミュレーション"], SUB)

    cy1, cy2 = row1 + bh / 2, row2 + bh / 2
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        _arrow(dr, (xs[a] + bw + 4, cy1), (xs[b] - 6, cy1))
    _arrow(dr, (xs[3] + bw / 2, row1 + bh + 4), (xs[3] + bw / 2, row2 - 6))
    _arrow(dr, (xs[3] - 4, cy2), (xs[2] + bw + 6, cy2))
    _arrow(dr, (xs[2] - 4, cy2), (xs[1] + bw + 6, cy2))
    # 失敗時の再試行ループ
    lx = xs[1] + bw / 2
    dr.line([(lx, row2 + bh + 4), (lx, row2 + bh + 40), (xs[3] + bw / 2, row2 + bh + 40)], fill=OK, width=3)
    _arrow(dr, (xs[3] + bw / 2, row2 + bh + 40), (xs[3] + bw / 2, row2 + bh + 6), OK)
    note = "失敗したら再認識からやり直し / 成功したら次の手順へ"
    dr.text((lx + 20, row2 + bh + 48), note, font=font(17), fill=OK)
    _arrow(dr, (xs[0] + bw + 4, cy2), (xs[1] - 6, cy2), SUB)
    dr.text((xs[0] + bw + 6, cy2 - 28), "画像", font=font(15), fill=SUB)
    foot = "対象物の位置はRGB-D画像から推定（シミュレータの真値はシーン生成・外乱の注入・評価にのみ使用）"
    fw = dr.textlength(foot, font=font(19))
    dr.text(((W - fw) / 2, H - 52), foot, font=font(19), fill=SUB)
    return img


def card(writer, llm_name: str, seconds: float):
    arr = np.asarray(render(llm_name)).astype(np.float32)
    n = int(seconds * FPS)
    for k in range(n):
        a = min(1.0, (k + 1) / 12, (n - k) / 8)
        writer.append_data((arr * a + np.array(BG) * (1 - a)).astype(np.uint8))
