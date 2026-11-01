"""デモ動画の生成: 4 つのシナリオを実行し、AI の内部状態を重ねた MP4 を書き出す。

使い方:
    python scripts/run_demo.py                      # 全シナリオ → outputs/kotoba_pick_demo.mp4
    python scripts/run_demo.py --scenarios 1 --fake-llm   # 動作確認用 (LLM を使わず固定計画)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import kotoba_pick  # noqa: E402,F401  (HF_HOME をプロジェクト内に設定)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from kotoba_pick.perception import Detector, Perception  # noqa: E402
from kotoba_pick.pipeline import Pipeline  # noqa: E402
from kotoba_pick.planner import LLMPlanner, Plan  # noqa: E402
from kotoba_pick.robot import Robot  # noqa: E402
from kotoba_pick.scene import build_scene, reset_robot  # noqa: E402
from kotoba_pick.video import Director, ja, open_writer, title_card  # noqa: E402
from kotoba_pick import diagram  # noqa: E402


@dataclasses.dataclass
class Scenario:
    key: str
    title: str
    subtitle: str
    instruction: str
    seed: int
    motion_speed: float = 1.0
    events: str = ""  # "disturb" で外乱を入れる
    fake_plan: list = dataclasses.field(default_factory=list)


SCENARIOS = [
    Scenario("1", "シナリオ1: 基本の指示", "色と形で対象を特定して運ぶ", "赤いブロックを木箱に入れて", seed=11,
             fake_plan=[("red cube", "wooden box")]),
    Scenario("2", "シナリオ2: 積み上げ", "物体の上面高さを深度から推定して精密に置く",
             "緑のブロックを青いブロックの上に積んで", seed=12, motion_speed=2.0,
             fake_plan=[("green cube", "blue cube")]),
    Scenario("3", "シナリオ3: 抽象的・複合的な指示", "「丸いもの」などの概念をLLMが解釈して複数手順に分解",
             "丸いものは全部お皿に、ブロックは全部木箱に片付けて", seed=13, motion_speed=2.0,
             fake_plan=[("red cube", "wooden box"), ("green cube", "wooden box"), ("blue cube", "wooden box"),
                        ("yellow ball", "white plate"), ("orange cylinder", "white plate")]),
    Scenario("4", "シナリオ4: 外乱からの回復", "途中で物を動かされても、落としても、再認識してやり直す",
             "ブロックを全部お皿に集めて", seed=14, motion_speed=2.0, events="disturb",
             fake_plan=[("red cube", "white plate"), ("green cube", "white plate"), ("blue cube", "white plate")]),
    Scenario("5", "シナリオ5: できない指示は断る", "存在しない物体を指示されたら、動かずに報告する",
             "紫のブロックを木箱に入れて", seed=15, fake_plan=[]),
]


class ScriptedPlanner:
    """--fake-llm 用: LLM を使わずに固定の計画を返す (動作確認のみに使用)。"""

    short_name = "scripted"

    def __init__(self):
        self.next = []

    def plan(self, instruction, objects, places, desc=None):
        return Plan([{"pick": a, "place": b} for a, b in self.next], "（テスト用の固定計画です）", source="llm")


# ---------------------------------------------------------------------------- 外乱
def _slide_object(pipe: Pipeline, body: str, to_xy, seconds: float = 1.0):
    """見えない「人の手」で物体をテーブル上で滑らせる (運動学的に位置を書き換え)。"""
    m, d = pipe.m, pipe.d
    jadr = m.jnt_qposadr[m.body(body).jntadr[0]]
    vadr = m.jnt_dofadr[m.body(body).jntadr[0]]
    start = d.qpos[jadr:jadr + 3].copy()
    goal = np.array([to_xy[0], to_xy[1], start[2]])
    n = int(seconds / m.opt.timestep)
    for k in range(1, n + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * k / n)
        d.qpos[jadr:jadr + 3] = start + s * (goal - start) + np.array([0, 0, 0.012 * np.sin(np.pi * k / n)])
        d.qvel[vadr:vadr + 6] = 0
        pipe.robot.step()
    pipe.robot.wait(0.3)


# 外乱の注入だけはシミュレータの内部情報 (物体の body 名) を使う。検知と回復はロボットが画像とグリッパ開度で自律的に行う。
LABEL2BODY = {"red cube": "red_cube", "green cube": "green_cube", "blue cube": "blue_cube",
              "yellow ball": "yellow_ball", "orange cylinder": "orange_can"}


def make_disturbances():
    state = {"slipped": False, "moved": False}

    def during_transport(idx, attempt, pipe: Pipeline, pick: str):
        # 最初の手順の搬送中に、いま掴んでいる物体の摩擦を一時的に下げて滑り落とす
        if state["slipped"] or pick not in LABEL2BODY:
            return
        state["slipped"] = True
        ui = pipe.ui
        ui.banner = f"外乱①: 搬送中に「{ja(pick)}」が滑り落ちる"
        ui.banner_sub = "(摩擦低下をスクリプトで注入。検知と回復はロボットが自律的に判断)"
        gid = pipe.m.geom(LABEL2BODY[pick] + "_geom").id
        fr = pipe.m.geom_friction[gid].copy()
        pipe.m.geom_friction[gid] = [0.05, 0.0, 0.0]
        for k in range(int(1.0 / pipe.m.opt.timestep)):
            ui.banner_alpha = min(1.0, k / 60)
            pipe.robot.step()
        pipe.m.geom_friction[gid] = fr
        ui.banner_alpha = 0.0
        ui.banner = None

    def before_step(idx, step, pipe: Pipeline):
        # 2 番目の手順に取りかかる直前に、人がその対象物を別の場所へ動かす
        pick = step["pick"]
        if idx != 1 or state["moved"] or pick not in LABEL2BODY:
            return
        state["moved"] = True
        ui = pipe.ui
        ui.banner = f"外乱②: 人が次の対象「{ja(pick)}」を動かす"
        ui.banner_sub = "(計画時とは違う場所へ。移動はスクリプトで注入)"
        for k in range(20):
            ui.banner_alpha = k / 20
            pipe.dir.emit()
        cur = pipe.d.xpos[pipe.m.body(LABEL2BODY[pick]).id]
        others = [pipe.d.xpos[pipe.m.body(b).id][:2] for lab, b in LABEL2BODY.items() if lab != pick]
        cands = [(0.60, 0.06), (0.60, -0.08), (0.36, 0.0), (0.62, 0.15), (0.62, -0.16)]
        # 現在地から十分離れ、他の物体とも重ならない場所を選ぶ
        to = max(cands, key=lambda c: min([np.hypot(c[0] - o[0], c[1] - o[1]) for o in others] +
                                          [0.5 * np.hypot(c[0] - cur[0], c[1] - cur[1])]))
        _slide_object(pipe, LABEL2BODY[pick], to, 1.2)
        for k in range(15):
            ui.banner_alpha = 1 - k / 15
            pipe.dir.emit()
        ui.banner = None

    return {"during_transport": during_transport, "before_step": before_step}


# ---------------------------------------------------------------------------- メイン
def run_scenario(sc: Scenario, writer, detector, planner, log_all: list):
    model, info = build_scene(seed=sc.seed)
    data = mujoco.MjData(model)
    reset_robot(model, data)
    director = Director(model, data, writer)
    perc_r = mujoco.Renderer(model, 720, 960)
    perception = Perception(model, perc_r, detector)
    robot = Robot(model, data, on_step=director.on_step)
    events = make_disturbances() if sc.events == "disturb" else {}
    pipe = Pipeline(model, data, robot, perception, planner, director, events)
    if isinstance(planner, ScriptedPlanner):
        planner.next = sc.fake_plan

    ui = director.ui
    ui.scenario = sc.title
    ui.plan_caption = ""
    # 状態表示を robot.status と連動させる
    orig_step = robot.step

    def step_with_status(n=1):
        ui.status = robot.status or ui.status
        orig_step(n)

    robot.step = step_with_status
    robot.wait(0.3)  # 物体を落ち着かせる (フレームは出力される)

    # シナリオ冒頭のバナー
    ui.banner, ui.banner_sub = sc.title, sc.subtitle

    def fade(p):
        ui.banner_alpha = min(1.0, p * 3, (1 - p) * 3)

    director.hold(1.8, fade)
    ui.banner = None

    # ロボットが動く区間だけ倍速 (AI の解説部分は等速)
    orig_move_joints, orig_move_linear = robot.move_joints, robot.move_linear

    def fast(fn):
        def wrapped(*a, **k):
            ui.speed = sc.motion_speed
            try:
                return fn(*a, **k)
            finally:
                ui.speed = 1.0
        return wrapped

    robot.move_joints, robot.move_linear = fast(orig_move_joints), fast(orig_move_linear)
    t0 = time.time()
    try:
        log = pipe.run(sc.instruction)
        director.hold(0.5)
    finally:
        director.close()
        perc_r.close()
    entry = {
        "scenario": sc.title, "instruction": sc.instruction, "seed": sc.seed,
        "plan": log.plan.steps if log.plan else None, "plan_source": log.plan.source if log.plan else None,
        "reply": log.plan.reply if log.plan else None, "think": getattr(log.plan, "think", ""),
        "llm_raw": log.plan.raw if log.plan else None,
        "plan_status": getattr(log.plan, "status", None), "plan_errors": getattr(log.plan, "errors", None),
        "results": [dataclasses.asdict(r) for r in log.results], "timings": log.timings,
        "status": log.status, "final_achieved": log.final_achieved,
        "sim_time": round(float(data.time), 1), "wall_time": round(time.time() - t0, 1),
    }
    log_all.append(entry)
    print(json.dumps({k: entry[k] for k in ("scenario", "plan", "results", "timings")}, ensure_ascii=False))
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "kotoba_pick_demo.mp4"))
    ap.add_argument("--scenarios", default="1,2,3,4,5")
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--llm-mode", default="decompose", choices=["decompose", "choice", "generate"])
    ap.add_argument("--no-cards", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    detector = Detector()
    planner = ScriptedPlanner() if args.fake_llm else LLMPlanner(args.llm, mode=args.llm_mode)
    print(f"models loaded in {time.time() - t_start:.1f}s")
    out = pathlib.Path(args.out)
    writer = open_writer(out)
    logs = []
    log_path = out.with_suffix(".json")
    llm_name = getattr(planner, "short_name", "LLM")
    try:
        _render_all(args, writer, detector, planner, llm_name, logs, log_path)
    finally:
        writer.close()
        log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"done: {out}  total wall time {time.time() - t_start:.0f}s")


def _render_all(args, writer, detector, planner, llm_name, logs, log_path):
    if not args.no_cards:
        title_card(writer, "KotobaPick", [
            "言葉で動くお片付けロボット",
            "日本語の指示 → ローカルLLMが手順を計画 → オープン語彙検出で対象を発見",
            "→ 深度で3D位置・向きを推定 → IKで把持・配置 → 画像で結果を確認",
            "AIはすべてCPUのみでローカル実行 (GPU・クラウドAPI不要)",
        ], 5.0, footer="※ AIの推論待ち時間は短縮して表示 (実際の推論時間は画面内に表示)。ロボット動作は等速/2倍速 (画面右上に表示)")
        diagram.card(writer, llm_name, 6.0)
    selected = [s for s in SCENARIOS if s.key in args.scenarios.split(",")]
    for sc in selected:
        run_scenario(sc, writer, detector, planner, logs)
        log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if not args.no_cards:
        lines = []
        for e in logs:
            n = len(e["results"])
            retries = sum(max(0, r["attempts"] - 1) for r in e["results"])
            if e["status"] == "no_plan":
                lines.append(f"{e['scenario'].split(':')[0]}: 計画なし（該当する物体なし）")
                continue
            lines.append(f"{e['scenario'].split(':')[0]}: 最終確認で {e['final_achieved']}/{n} 手順を達成" +
                         (f"（途中の再試行 {retries} 回）" if retries else ""))
        det_t = [t for e in logs for t in e["timings"].get("detector", [])]
        llm_t = [t for e in logs for t in e["timings"].get("llm", [])]
        if det_t and llm_t:
            lines.append(f"実際の推論時間 (CPU): 物体検出 平均{np.mean(det_t):.1f}秒 / "
                         f"LLM計画 平均{np.mean(llm_t):.0f}秒")
        title_card(writer, "結果", lines + ["MuJoCo + Franka Panda / Grounding DINO / " + llm_name],
                   5.0, footer="KotobaPick — Physical AI 応用講座 最終課題")


if __name__ == "__main__":
    main()
