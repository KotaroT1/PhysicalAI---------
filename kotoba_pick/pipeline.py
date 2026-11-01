"""統合パイプライン: 言語指示 → シーン認識 → LLM 計画 → (再認識 → 把持 → 配置 → 結果確認) × ステップ数。

各ステップの前に必ずカメラで再認識し、配置後も画像で結果を確かめる閉ループ構成。
人が途中で物を動かしても、把持に失敗しても、再認識してやり直せる。
"""
from __future__ import annotations

import dataclasses
import re
import time
from typing import Callable, Optional

import mujoco
import numpy as np

from .perception import Detection, Perception, SceneState, describe_position
from .planner import Plan
from .robot import Robot
from .video import ACCENT, NG, OK, WARN, Director, ja

GRIPPER_HALF_SPAN = 0.05  # 指を開いたときに容器の壁と干渉しない余裕 [m]


@dataclasses.dataclass
class StepResult:
    pick: str
    place: str
    success: bool
    attempts: int
    note: str = ""
    height: float = 0.044  # 把持時に推定した物体の高さ [m] (結果確認に使用)
    base: Optional[list] = None  # 積み上げ時の土台の [x, y, 上面高さ] (置く前に計測)
    final_ok: Optional[bool] = None  # 全手順終了後の最終確認の結果


@dataclasses.dataclass
class EpisodeLog:
    instruction: str
    plan: Optional[Plan] = None
    results: list = dataclasses.field(default_factory=list)
    timings: dict = dataclasses.field(default_factory=dict)  # AI 推論の実時間 [s]
    status: str = ""  # "done" | "no_plan"
    final_achieved: int = 0

    def add_time(self, key, sec):
        self.timings.setdefault(key, []).append(round(sec, 2))


def _confirm_text(steps: list) -> str:
    """計画を日本語の確認文にする。例:「黄色いボール・オレンジの缶を白い皿に、青いブロックを木箱に運びます。」"""
    groups: dict = {}
    for st in steps:
        groups.setdefault(st["place"], []).append(st["pick"])
    parts, stack_only = [], True
    for place, picks in groups.items():
        names = "・".join(ja(p) for p in picks)
        if place in ("wooden box", "white plate"):
            parts.append(f"{names}を{ja(place)}に")
            stack_only = False
        else:
            parts.append(f"{names}を{ja(place)}の上に")
    return "、".join(parts) + ("積みます。" if stack_only else "運びます。")


def _ja_reason(err: str) -> str:
    m = re.match(r'no object matches "(.+)"', err)
    if m:
        return f"「{m.group(1)}」に当てはまる物体がありません"
    m = re.match(r'destination "(.+)" not found', err)
    if m:
        return f"置き場所「{m.group(1)}」が見つかりません"
    return err


class Pipeline:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot: Robot, perception: Perception,
                 planner, director: Optional[Director] = None, events: Optional[dict] = None):
        self.m, self.d = model, data
        self.robot = robot
        self.per = perception
        self.planner = planner
        self.dir = director
        self.events = events or {}  # デモ用の外乱 {"before_step": fn(i, step, pipe), "during_transport": fn(i, attempt, pipe, pick)}
        self._containers: dict = {}  # 結果確認用に直近の容器の検出結果を保持

    # ------------------------------------------------------------------ 演出ヘルパ
    @property
    def ui(self):
        return self.dir.ui if self.dir else None

    def _hold(self, sec, anim=None):
        if self.dir:
            self.dir.hold(sec, anim)

    def _caption(self, text, color=ACCENT):
        if self.dir:
            self.ui.caption, self.ui.caption_color = text, color

    def _show_scene(self, st: SceneState, caption: str, highlight=None, reveal_sec: float = 1.2):
        if not self.dir:
            return
        self.ui.perc_img, self.ui.perc_dets, self.ui.highlight = st.rgb, st.detections, highlight
        self.ui.perc_caption = caption

        def anim(p):
            self.ui.perc_reveal = p

        self._hold(reveal_sec, anim)

    # ------------------------------------------------------------------ 認識
    def perceive(self, log: EpisodeLog, highlight=None, reveal_sec=1.0) -> SceneState:
        if self.dir:
            self.ui.status = "カメラで認識中"
        st = self.per.observe(self.d)
        log.add_time("detector", st.seconds)
        self._show_scene(st, f"ロボットの目 (Grounding DINO / CPU {st.seconds:.1f}秒)", highlight, reveal_sec)
        return st

    # ------------------------------------------------------------------ メイン
    def run(self, instruction: str) -> EpisodeLog:
        log = EpisodeLog(instruction)
        ui = self.ui
        if ui:
            ui.instruction, ui.reply, ui.plan, ui.step_status, ui.markers = instruction, "", [], [], []
            ui.plan_caption, ui.status = "", "待機中"
            self._caption("ユーザーからの指示を受信")

            def type_anim(p):
                ui.instr_reveal = p

            self._hold(1.2, type_anim)

        # 1) シーン認識 (何が机の上にあるか)
        self._caption("① シーン認識: 机の上の物体をオープン語彙検出 + 深度で3D位置を推定")
        st = self.perceive(log, reveal_sec=1.2)
        objects = [d.label for d in st.objects]
        places = [d.label for d in st.containers]
        desc = {}
        for d in st.objects:
            inside = self._inside_container(d, st)
            desc[d.label] = f"in {inside}" if inside else describe_position(d.center)

        # 2) LLM による計画
        self._caption("② タスク計画: ローカルLLMが指示を「何を→どこへ」に構造化し、対象の物体を特定")
        if ui:
            ui.status = "LLMで計画中"
        plan = self.planner.plan(instruction, objects, places, desc)
        log.plan = plan
        log.add_time("llm", plan.seconds)
        if ui:
            ui.plan = [(s["pick"], s["place"]) for s in plan.steps]
            ui.step_status = ["pending"] * len(plan.steps)
            src = {"llm": "", "llm-retry": " / 自己修正で再生成", "rule": " / ルールにフォールバック"}[plan.source]
            name = getattr(self.planner, "short_name", "LLM")
            mode = getattr(self.planner, "mode", "")
            if mode == "decompose" and plan.think:
                how = "理解: " + plan.think  # LLM が構造化した「何を → どこへ」
            elif mode == "choice":
                how = "物体ごとに行き先の候補をLLMが採点して選択"
            else:
                how = "出力をシーンと照合して検証済み"
            ui.plan_caption = f"{name} (CPU {plan.seconds:.0f}秒{src}) {how}"
            full_plan = list(ui.plan)

            def plan_anim(p):
                ui.plan = full_plan[: int(np.ceil(len(full_plan) * p))]

            self._hold(max(1.0, 0.4 * len(full_plan)), plan_anim)
            ui.plan = full_plan
            # 小型 LLM の返答文は文法が乱れることがあるため、画面には計画から組み立てた確認文を出す (LLM の返答はログに残す)
            ui.reply = _confirm_text(plan.steps) if plan.steps else plan.reply
            self._hold(0.8)

        # 計画が立たなかった (該当する物体がない・不正な計画) 場合は何もしない
        if not plan.steps:
            reason = "、".join(_ja_reason(e) for e in plan.errors) if plan.errors else "該当する物体がありません"
            log.status = "no_plan"
            if ui:
                ui.status = "待機中"
                # LLM の返答文は対象の有無や計画の正しさを知らずに作られるため、固定の報告文に差し替える
                ui.reply = ("ご指示の物体が見つからないため、作業は行いません。" if getattr(plan, "status", "") == "empty"
                            else "実行できる計画を立てられなかったため、作業は行いません。")
                self._caption(f"△ 実行できる手順がありません（{reason}）→ 動かずに報告", WARN)
                self._hold(2.5)
            return log

        if plan.errors and ui:  # 一部の対象が見つからない場合は、その旨を伝えてから残りを実行する
            reason = "、".join(_ja_reason(e) for e in plan.errors)
            self._caption(f"△ {reason} → 見つかった物体だけ実行します", WARN)
            self._hold(1.5)

        # 3) 実行
        failed = set()
        for i, step in enumerate(plan.steps):
            if ui:
                ui.step_status[i] = "active"
            if step["place"] in failed:  # 土台にする物体の移動に失敗したので積めない
                res = StepResult(step["pick"], step["place"], False, 0, "skipped: base object failed")
                self._caption(f"△ 土台の「{ja(step['place'])}」が未完了のためスキップ", WARN)
                self._hold(1.0)
            else:
                if "before_step" in self.events:
                    self.events["before_step"](i, step, self)
                res = self.execute_step(step["pick"], step["place"], log, i)
            log.results.append(res)
            if not res.success:
                failed.add(step["pick"])
            if ui:
                ui.step_status[i] = "done" if res.success else ("skip" if res.attempts == 0 else "failed")
                ui.markers = []

        # 4) 最終確認: 後の手順で先に置いた物体を動かしていないか、全結果をもう一度画像で確かめる
        self._caption("⑦ 最終確認: すべての手順の結果をもう一度カメラで確認")
        st = self.perceive(log, reveal_sec=0.6)
        for i, res in enumerate(log.results):
            if res.success:
                res.final_ok = self._verify(res.pick, res.place, st, res.height, res.base)
                if ui and not res.final_ok:
                    ui.step_status[i] = "failed"
        log.final_achieved = sum(bool(r.final_ok) for r in log.results)
        log.status = "done"
        n = len(log.results)
        if ui:
            ui.status = "完了"
            self._caption(f"タスク完了: 最終確認で {log.final_achieved}/{n} 手順の達成を確認",
                          OK if log.final_achieved == n else WARN)
            self._hold(1.2)
        return log

    # ------------------------------------------------------------------ 1 ステップ
    def execute_step(self, pick: str, place: str, log: EpisodeLog, idx: int, max_attempts: int = 3) -> StepResult:
        ui = self.ui
        note = ""
        height = 0.044
        for attempt in range(1, max_attempts + 1):
            self._caption(f"③ 再認識: 「{ja(pick)}」と「{ja(place)}」の現在位置を確認"
                          + (f" (再試行 {attempt - 1})" if attempt > 1 else ""))
            st = self.perceive(log, highlight=pick, reveal_sec=0.6)
            obj, tgt = st.get(pick), st.get(place)
            if obj is None or tgt is None:
                note = f"{ja(pick) if obj is None else ja(place)} が見つかりません"
                if ui:
                    ui.step_status[idx] = "retry"
                self._caption("× " + note + (" → もう一度探します" if attempt < max_attempts else ""), NG)
                self._hold(1.0)
                continue
            if tgt.is_container and self._inside_container(obj, st) == place and not self._on_other(obj, st):
                self._caption(f"「{ja(pick)}」はすでに{ja(place)}にあります → スキップ", OK)
                self._hold(1.0)
                return StepResult(pick, place, True, attempt, "already there", obj.top_z - self._support_z(obj, st))

            grasp, yaw = self._grasp_pose(obj)
            support = self._support_z(obj, st)
            height = obj.top_z - support
            place_pos = self._place_pose(obj, tgt, st, grasp[2] - support)
            base = None if tgt.is_container else [float(tgt.center[0]), float(tgt.center[1]), float(tgt.top_z)]
            if tgt.is_container:
                self._containers[place] = tgt
            yaw = self.robot.choose_yaw(grasp, yaw, np.pi / 2)
            if ui:
                ui.markers = [(grasp, (0.2, 0.7, 1.0, 0.55), 0.012), (place_pos, (0.3, 1.0, 0.5, 0.55), 0.012)]
            self._caption(f"④ 把持: 推定位置 ({grasp[0]:.3f}, {grasp[1]:.3f}, {grasp[2]:.3f}) m, "
                          f"向き {np.rad2deg(yaw):.0f}° (深度画像から推定)")
            ok = self.robot.pick(grasp, yaw)
            if not ok:
                if ui:
                    ui.step_status[idx] = "retry"
                self._caption("× 把持失敗を検知 (グリッパ開度) → 再認識してやり直し", NG)
                self.robot.go_observe()
                note = "grasp failed"
                continue
            self._caption(f"⑤ 配置: 「{ja(place)}」の" + ("空いている場所へ" if tgt.is_container else "上に積む"))
            if "during_transport" in self.events:
                self.events["during_transport"](idx, attempt, self, pick)
            place_yaw = self.robot.choose_yaw(place_pos, yaw, np.pi / 2)
            if not self.robot.holding() or not self.robot.place(place_pos, place_yaw):
                if ui:
                    ui.step_status[idx] = "retry"
                    ui.markers = []
                self._caption("× 搬送中の落下を検知 (グリッパ開度) → 再認識してやり直し", NG)
                self._hold(1.0)
                self.robot.go_observe()
                note = "dropped during transport"
                continue
            self.robot.go_observe()
            if ui:
                ui.markers = []
            # 6) 結果の確認 (閉ループ)
            self._caption("⑥ 結果確認: カメラ画像で配置できたかを検証")
            st2 = self.perceive(log, highlight=pick, reveal_sec=0.5)
            if self._verify(pick, place, st2, height, base):
                self._caption(f"○ 「{ja(pick)}」を{ja(place)}" + ("に配置できました" if tgt.is_container
                                                                  else "の上に積めました"), OK)
                self._hold(0.6)
                return StepResult(pick, place, True, attempt, note, height, base)
            if ui:
                ui.step_status[idx] = "retry"
            note = "verification failed"
            self._caption("× 結果確認で失敗を検知 → もう一度やり直します", NG)
            self._hold(1.0)
        return StepResult(pick, place, False, max_attempts, note, height, None)

    # ------------------------------------------------------------------ 幾何計算
    @staticmethod
    def _grasp_pose(obj: Detection):
        top = obj.top_z
        grasp_z = max(0.012, top - min(0.022, top / 2))  # 上面から最大 2.2cm 下を掴む
        yaw = obj.yaw if "cube" in obj.label else 0.0
        return np.array([obj.center[0], obj.center[1], grasp_z]), yaw

    @staticmethod
    def _on_other(obj: Detection, st: SceneState) -> bool:
        """他の物体の上に乗っているか (真下に、上面がより低い物体があるか)。"""
        return any(d.label != obj.label and np.linalg.norm(d.center[:2] - obj.center[:2]) < 0.03
                   and d.top_z < obj.top_z - 0.02 for d in st.objects)

    def _support_z(self, obj: Detection, st: SceneState) -> float:
        """物体を支えている面の高さ: 容器の底面 / 下にある物体の上面 / テーブル (0)。"""
        c = self._inside_container(obj, st)
        if c is not None:
            return st.get(c).surface_z
        below = [d.top_z for d in st.objects if d.label != obj.label
                 and np.linalg.norm(d.center[:2] - obj.center[:2]) < 0.03 and d.top_z < obj.top_z - 0.02]
        return max(below) if below else 0.0

    def _place_pose(self, obj: Detection, tgt: Detection, st: SceneState, below: float):
        """below: 把持点から物体の底面までの距離 [m]。"""
        if not tgt.is_container:
            return np.array([tgt.center[0], tgt.center[1], tgt.top_z + below + 0.006])
        cx, cy = tgt.center[:2]
        if "plate" in tgt.label:
            offs = [(0, 0), (0.0, 0.065), (0.0, -0.065), (0.065, 0.0), (-0.065, 0.0)]
        else:
            offs = [(0, 0), (0.05, 0.05), (-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05)]
        occupied = [d.center[:2] for d in st.objects
                    if d.label != obj.label and self._inside_container(d, st) == tgt.label]
        best, best_score = offs[0], -np.inf
        for k, (dx, dy) in enumerate(offs):
            q = np.array([cx + dx, cy + dy])
            clear = min([np.linalg.norm(q - o) for o in occupied], default=1.0)
            score = min(clear, 0.09) - 0.001 * k  # 空いていて、なるべく中央寄り
            if score > best_score:
                best, best_score = (dx, dy), score
        return np.array([cx + best[0], cy + best[1], tgt.surface_z + below + 0.005])  # 低い位置で離して転がりを防ぐ

    @staticmethod
    def _inside(d: Detection, c: Detection, margin: float = 0.015) -> bool:
        """物体中心が容器の内側にあるか (皿は円、箱は矩形として判定)。"""
        if c is None or c.extent is None:
            return False
        x0, x1, y0, y1 = c.extent
        if "plate" in c.label:
            r = min(x1 - x0, y1 - y0) / 2
            return float(np.hypot(d.center[0] - (x0 + x1) / 2, d.center[1] - (y0 + y1) / 2)) < r - margin
        return x0 + margin < d.center[0] < x1 - margin and y0 + margin < d.center[1] < y1 - margin

    def _inside_container(self, d: Detection, st: SceneState) -> Optional[str]:
        for c in st.containers:
            if self._inside(d, c):
                return c.label
        return None

    def _verify(self, pick: str, place: str, st: SceneState, height: float, base: Optional[list] = None) -> bool:
        """置けたかの判定。位置 (容器の内側 / 土台の真上) に加えて高さも見て、縁や他の物体への乗り上げを除外する。
        積み上げは、上に物体が載ると土台の検出が乱れるため、置く前に測った土台の位置・高さ (base) と比べる。"""
        obj = st.get(pick)
        if obj is None:
            return False
        if base is None:
            cont = st.get(place) or self._containers.get(place)
            if cont is None:
                return False
            return (self._inside(obj, cont) and obj.top_z - height < cont.surface_z + 0.015
                    and not self._on_other(obj, st))
        dxy = float(np.hypot(obj.center[0] - base[0], obj.center[1] - base[1]))
        return dxy < 0.025 and abs(obj.top_z - (base[2] + height)) < 0.015
