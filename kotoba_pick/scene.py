"""MuJoCo シーン構築: Franka Panda + テーブル + 操作対象物 + 容器 + カメラ。

Windows では MuJoCo が非 ASCII パス (日本語フォルダ名) を開けないため、
MJCF とメッシュをすべてメモリに読み込んでから MjSpec に渡している。
"""
from __future__ import annotations

import dataclasses
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
PANDA_DIR = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"

# ロボットの「観察姿勢」: カメラ視野から腕を退避させた姿勢 (関節角 rad)
OBSERVE_Q = np.array([0.0, -0.6, 0.0, -2.6, 0.0, 2.0, -0.785])
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
GRIPPER_RELEASE = 200.0  # 物体を離すときは開き幅 6.3cm に抑え、隣の物体に指を当てない


@dataclasses.dataclass
class ObjectSpec:
    """操作対象物の定義。name はシミュレータ内部の ID (検証用のみ。AI には見せない)。"""

    name: str
    shape: str  # "box" | "sphere" | "cylinder"
    size: tuple  # MuJoCo の geom size (half-size)
    rgba: tuple
    mass: float = 0.06

    @property
    def half_height(self) -> float:
        if self.shape == "box":
            return self.size[2]
        if self.shape == "sphere":
            return self.size[0]
        return self.size[1]  # cylinder: (radius, half-height)


@dataclasses.dataclass
class ContainerSpec:
    name: str
    kind: str  # "box" (木箱) | "plate" (皿)
    pos: tuple  # テーブル上の中心 (x, y)
    rgba: tuple
    inner: float = 0.085  # 内寸の半分 (box) / 半径 (plate)
    wall_h: float = 0.045


DEFAULT_OBJECTS = [
    ObjectSpec("red_cube", "box", (0.022, 0.022, 0.022), (0.85, 0.12, 0.10, 1)),
    ObjectSpec("green_cube", "box", (0.022, 0.022, 0.022), (0.15, 0.65, 0.20, 1)),
    ObjectSpec("blue_cube", "box", (0.022, 0.022, 0.022), (0.12, 0.30, 0.85, 1)),
    ObjectSpec("yellow_ball", "sphere", (0.024,), (0.95, 0.80, 0.10, 1), mass=0.05),
    ObjectSpec("orange_can", "cylinder", (0.022, 0.032), (0.95, 0.45, 0.08, 1), mass=0.06),
]

DEFAULT_CONTAINERS = [
    ContainerSpec("wooden_box", "box", (0.40, 0.36), (0.55, 0.36, 0.18, 1), inner=0.10),
    ContainerSpec("white_plate", "plate", (0.43, -0.35), (0.96, 0.96, 0.94, 1), inner=0.11),
]


def _load_panda_spec() -> mujoco.MjSpec:
    xml = (PANDA_DIR / "panda.xml").read_text(encoding="utf-8")
    assets = {"assets/" + f.name: f.read_bytes() for f in (PANDA_DIR / "assets").iterdir()}
    spec = mujoco.MjSpec.from_string(xml, assets=assets)
    for key in list(spec.keys):  # 物体を追加すると qpos 長が変わるため keyframe は削除
        spec.delete(key)
    return spec


def _add_container(world, c: ContainerSpec):
    body = world.add_body(name=c.name, pos=[c.pos[0], c.pos[1], 0.0])
    if c.kind == "plate":
        body.add_geom(name=c.name + "_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[c.inner, 0.006, 0], pos=[0, 0, 0.006], rgba=c.rgba, friction=[1.0, 0.01, 0.001])
        body.add_geom(name=c.name + "_rim", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[c.inner + 0.012, 0.003, 0], pos=[0, 0, 0.003],
                      rgba=[c.rgba[0] * 0.85, c.rgba[1] * 0.85, c.rgba[2] * 0.85, 1])
        return
    t = 0.008  # 壁厚
    h = c.wall_h
    s = c.inner
    body.add_geom(name=c.name + "_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[s + t, s + t, t / 2], pos=[0, 0, t / 2], rgba=c.rgba)
    dark = [c.rgba[0] * 0.8, c.rgba[1] * 0.8, c.rgba[2] * 0.8, 1]
    for i, (dx, dy, sx, sy) in enumerate([(s + t / 2, 0, t / 2, s + t), (-(s + t / 2), 0, t / 2, s + t),
                                          (0, s + t / 2, s, t / 2), (0, -(s + t / 2), s, t / 2)]):
        body.add_geom(name=f"{c.name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[sx, sy, h / 2], pos=[dx, dy, h / 2], rgba=dark if i % 2 else c.rgba)


def _add_object(world, o: ObjectSpec, pos, yaw: float):
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
    body = world.add_body(name=o.name, pos=[pos[0], pos[1], o.half_height + 0.001], quat=quat)
    body.add_freejoint(name=o.name + "_free")
    gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
             "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}[o.shape]
    size = list(o.size) + [0] * (3 - len(o.size))
    body.add_geom(name=o.name + "_geom", type=gtype, size=size, rgba=o.rgba, mass=o.mass,
                  condim=6 if o.shape == "sphere" else 4,
                  friction=[1.5, 0.01, 0.002], solref=[0.005, 1], priority=1)


def sample_layout(rng: np.random.Generator, objects, n_try: int = 2000, min_dist: float = 0.085):
    """物体同士が重ならない初期配置をランダムに生成する。"""
    poses = []
    for _ in objects:
        for _ in range(n_try):
            p = np.array([rng.uniform(0.36, 0.62), rng.uniform(-0.17, 0.17)])
            if all(np.linalg.norm(p - q[:2]) > min_dist for q in poses):
                poses.append(np.array([p[0], p[1], rng.uniform(-0.25, 0.25)]))
                break
        else:
            raise RuntimeError("配置のサンプリングに失敗しました")
    return poses


def build_scene(seed: int = 0, objects=None, containers=None, layout=None,
                render_size=(1280, 720)) -> tuple[mujoco.MjModel, dict]:
    objects = DEFAULT_OBJECTS if objects is None else objects
    containers = DEFAULT_CONTAINERS if containers is None else containers
    rng = np.random.default_rng(seed)
    layout = sample_layout(rng, objects) if layout is None else layout

    spec = _load_panda_spec()
    spec.option.timestep = 0.002
    spec.option.noslip_iterations = 3  # 把持中の滑りを抑える
    spec.visual.global_.offwidth = max(render_size[0], 1280)
    spec.visual.global_.offheight = max(render_size[1], 960)
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.22, 0.22, 0.22]
    spec.visual.headlight.diffuse = [0.30, 0.30, 0.30]

    # ロボットは重力補償付き (実機 Panda の制御器と同様) にして位置追従誤差をなくす
    for body in spec.bodies:
        if body.name and body.name != "world":
            body.gravcomp = 1.0

    # TCP (指先中心) サイト
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.004, 0, 0], rgba=[1, 0, 0, 0])

    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.22, 0.24, 0.27], rgb2=[0.18, 0.20, 0.23], width=512, height=512)
    spec.add_material(name="floor", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.1)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.42, 0.52, 0.62], rgb2=[0.08, 0.10, 0.14], width=512, height=3072)

    world = spec.worldbody
    for light in list(spec.lights):
        spec.delete(light)
    world.add_light(name="key", pos=[0.8, -0.6, 2.2], dir=[-0.3, 0.25, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[0.50, 0.50, 0.50], specular=[0.15, 0.15, 0.15], castshadow=True)
    world.add_light(name="fill", pos=[-0.5, 0.8, 2.0], dir=[0.3, -0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[0.20, 0.20, 0.20], castshadow=False)

    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], pos=[0, 0, -0.75],
                   material="floor")
    # テーブル (上面 z=0)
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.55, 0.62, 0.02], pos=[0.40, 0, -0.02],
                   rgba=[0.42, 0.45, 0.50, 1], friction=[1.0, 0.01, 0.001])
    for i, (lx, ly) in enumerate([(-0.10, -0.57), (-0.10, 0.57), (0.90, -0.57), (0.90, 0.57)]):
        world.add_geom(name=f"leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.365],
                       pos=[lx, ly, -0.405], rgba=[0.35, 0.33, 0.30, 1])

    for c in containers:
        _add_container(world, c)
    for o, p in zip(objects, layout):
        _add_object(world, o, p[:2], p[2])

    # 認識用カメラ: ロボットの正面上方からテーブルを見下ろす (ロボットの目)
    _add_lookat_camera(world, "perception", eye=[0.98, 0.0, 0.72], target=[0.44, 0.0, 0.0], fovy=58)
    # 動画用カメラ
    _add_lookat_camera(world, "overview", eye=[1.25, -0.78, 0.95], target=[0.32, 0.02, 0.22], fovy=44)
    _add_lookat_camera(world, "side", eye=[0.45, -1.05, 0.45], target=[0.45, 0.0, 0.05], fovy=45)

    model = spec.compile()
    info = {"objects": objects, "containers": containers, "layout": layout}
    return model, info


def _add_lookat_camera(world, name, eye, target, fovy):
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    # MuJoCo カメラは -z 方向を向き、x が右、y が上
    world.add_camera(name=name, pos=eye, xyaxes=np.concatenate([right, cam_up]), fovy=fovy)


def reset_robot(model: mujoco.MjModel, data: mujoco.MjData, q=OBSERVE_Q):
    for i in range(7):
        jid = model.joint(f"joint{i + 1}").id
        data.qpos[model.jnt_qposadr[jid]] = q[i]
        data.ctrl[i] = q[i]
    for fj in ("finger_joint1", "finger_joint2"):
        data.qpos[model.jnt_qposadr[model.joint(fj).id]] = 0.04
    data.ctrl[7] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)


def object_pos(model, data, name: str) -> np.ndarray:
    return data.xpos[model.body(name).id].copy()
