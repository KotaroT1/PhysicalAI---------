"""Franka Panda の動作生成: 減衰最小二乗法 IK + 最小躍度軌道 + 把持プリミティブ。"""
from __future__ import annotations

from typing import Callable, Optional

import mujoco
import numpy as np

from .scene import GRIPPER_CLOSED, GRIPPER_OPEN, GRIPPER_RELEASE, OBSERVE_Q

# 姿勢優先度 (冗長自由度を自然な姿勢に寄せるための目標)
REST_Q = np.array([0.0, -0.5, 0.0, -2.3, 0.0, 1.85, 0.785])

APPROACH_H = 0.12  # 把持前に対象物の上で待機する高さ [m]
LIFT_H = 0.16  # 持ち上げ高さ [m]


def _min_jerk(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def grasp_rotation(yaw: float) -> np.ndarray:
    """指先が真下を向き、指の開閉方向が world の (−sin yaw, cos yaw) となる回転行列。"""
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    r0 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)  # x=-X, y=+Y, z=-Z
    return rz @ r0


class Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 on_step: Optional[Callable[[], None]] = None):
        self.m, self.d = model, data
        self.on_step = on_step  # 1 制御ステップごとに呼ばれる (動画記録用)
        self.ik_d = mujoco.MjData(model)
        self.qadr = np.array([model.jnt_qposadr[model.joint(f"joint{i + 1}").id] for i in range(7)])
        self.dadr = np.array([model.jnt_dofadr[model.joint(f"joint{i + 1}").id] for i in range(7)])
        self.lo = np.array([model.jnt_range[model.joint(f"joint{i + 1}").id][0] for i in range(7)])
        self.hi = np.array([model.jnt_range[model.joint(f"joint{i + 1}").id][1] for i in range(7)])
        self.tcp = model.site("tcp").id
        self.f1 = model.jnt_qposadr[model.joint("finger_joint1").id]
        self.f2 = model.jnt_qposadr[model.joint("finger_joint2").id]
        self.q_cmd = data.ctrl[:7].copy()
        self.grip_cmd = float(data.ctrl[7])
        self.status = ""  # 動画オーバーレイ用の現在動作ラベル

    # ------------------------------------------------------------------ 基本
    def step(self, n: int = 1):
        for _ in range(n):
            self.d.ctrl[:7] = self.q_cmd
            self.d.ctrl[7] = self.grip_cmd
            mujoco.mj_step(self.m, self.d)
            if self.on_step is not None:
                self.on_step()

    def wait(self, seconds: float):
        self.step(int(round(seconds / self.m.opt.timestep)))

    def tcp_pose(self, d: Optional[mujoco.MjData] = None):
        d = self.d if d is None else d
        return d.site_xpos[self.tcp].copy(), d.site_xmat[self.tcp].reshape(3, 3).copy()

    def gripper_width(self) -> float:
        return float(self.d.qpos[self.f1] + self.d.qpos[self.f2])

    # ------------------------------------------------------------------ IK
    def solve_ik(self, pos, rot, q_init=None, iters: int = 300, tol: float = 1e-4):
        """減衰最小二乗法 + ゼロ空間での姿勢誘導による 6 自由度 IK。"""
        d = self.ik_d
        d.qpos[:] = self.d.qpos
        q = (self.q_cmd if q_init is None else q_init).copy()
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        lam = 0.05
        err_norm = np.inf
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p_cur = d.site_xpos[self.tcp]
            r_cur = d.site_xmat[self.tcp].reshape(3, 3)
            e_pos = pos - p_cur
            e_rot = 0.5 * sum(np.cross(r_cur[:, i], rot[:, i]) for i in range(3))
            err = np.concatenate([e_pos, e_rot])
            err_norm = np.linalg.norm(err)
            if err_norm < tol:
                break
            mujoco.mj_jacSite(self.m, d, jacp, jacr, self.tcp)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
            JJt = J @ J.T + lam**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            # ゼロ空間で自然な姿勢へ誘導
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (0.1 * (REST_Q - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo + 1e-3, self.hi - 1e-3)
        return q, err_norm

    # ------------------------------------------------------------------ 軌道
    def move_joints(self, q_goal, duration: float):
        q0 = self.q_cmd.copy()
        n = max(1, int(round(duration / self.m.opt.timestep)))
        for k in range(1, n + 1):
            s = _min_jerk(k / n)
            self.q_cmd = q0 + s * (q_goal - q0)
            self.step()

    def move_linear(self, pos_goal, rot, duration: float, ik_every: int = 5):
        """TCP を直線で移動 (数ステップごとに IK を解き、間は関節空間で補間)。"""
        p0, _ = self.tcp_pose()
        p0 = self._tcp_of(self.q_cmd)
        n = max(1, int(round(duration / self.m.opt.timestep)))
        q_prev = self.q_cmd.copy()
        k = 0
        while k < n:
            k2 = min(n, k + ik_every)
            s = _min_jerk(k2 / n)
            q_next, _ = self.solve_ik(p0 + s * (pos_goal - p0), rot, q_init=q_prev, iters=50)
            for j in range(k + 1, k2 + 1):
                a = (j - k) / (k2 - k)
                self.q_cmd = q_prev + a * (q_next - q_prev)
                self.step()
            q_prev = q_next
            k = k2

    def _tcp_of(self, q):
        d = self.ik_d
        d.qpos[:] = self.d.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.tcp].copy()

    def set_gripper(self, value: float, duration: float = 0.6):
        g0 = self.grip_cmd
        n = max(1, int(round(duration / self.m.opt.timestep)))
        for k in range(1, n + 1):
            self.grip_cmd = g0 + (value - g0) * min(1.0, 2.0 * k / n)
            self.step()

    def _duration_for(self, q_goal, vmax: float = 1.1, tmin: float = 0.9) -> float:
        return max(tmin, float(np.max(np.abs(q_goal - self.q_cmd))) / vmax * 1.875)

    # ------------------------------------------------------------------ プリミティブ
    def go_observe(self):
        self.status = "観察姿勢へ移動"
        self.move_joints(OBSERVE_Q, self._duration_for(OBSERVE_Q))

    def choose_yaw(self, pos, base_yaw: float, symmetry: float = np.pi / 2) -> float:
        """対称性を考慮し、IK が楽な (関節7 が中央寄りの) 把持ヨー角を選ぶ。"""
        best, best_cost = base_yaw, np.inf
        for k in range(int(round(2 * np.pi / symmetry))):
            yaw = base_yaw + k * symmetry
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            q, err = self.solve_ik(pos, grasp_rotation(yaw), q_init=REST_Q)
            cost = err * 100 + abs(q[6] - REST_Q[6]) + 0.3 * np.linalg.norm(q - REST_Q)
            if cost < best_cost:
                best, best_cost = yaw, cost
        return best

    def pick(self, pos, yaw: float = 0.0) -> bool:
        """pos (把持点 world 座標) に上から接近して掴み、持ち上げる。成功判定はグリッパ開度で行う。"""
        rot = grasp_rotation(yaw)
        above = pos + np.array([0, 0, APPROACH_H])
        self.status = "対象物の上へ移動"
        self.set_gripper(GRIPPER_OPEN, 0.2)
        q_above, _ = self.solve_ik(above, rot, q_init=REST_Q)
        self.move_joints(q_above, self._duration_for(q_above))
        self.status = "下降"
        self.move_linear(pos, rot, 0.9)
        self.status = "把持"
        self.set_gripper(GRIPPER_CLOSED, 0.7)
        self.status = "持ち上げ"
        self.move_linear(pos + np.array([0, 0, LIFT_H]), rot, 0.8)
        return self.holding()  # 指が閉じきっていなければ何かを掴んでいる

    def holding(self) -> bool:
        return self.gripper_width() > 0.01

    def place(self, pos, yaw: float = 0.0) -> bool:
        """pos (物体を離す TCP 位置) へ運んで離す。搬送中に物体を落としたら False を返す。"""
        rot = grasp_rotation(yaw)
        above = pos + np.array([0, 0, APPROACH_H])
        self.status = "目標位置へ搬送"
        q_above, _ = self.solve_ik(above, rot)
        self.move_joints(q_above, self._duration_for(q_above, vmax=0.9))
        if not self.holding():
            return False
        self.status = "下降"
        self.move_linear(pos, rot, 0.8)
        self.status = "解放"
        self.set_gripper(GRIPPER_RELEASE, 0.5)
        self.status = "退避"
        self.move_linear(pos + np.array([0, 0, APPROACH_H]), rot, 0.6)
        return True
