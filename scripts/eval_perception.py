"""知覚モジュールの精度評価: ランダム配置のシーンで、検出ラベルと推定 3D 位置を真値と比較する。

    python scripts/eval_perception.py --seeds 10
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import kotoba_pick  # noqa: E402,F401
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from kotoba_pick.perception import Detector, Perception  # noqa: E402
from kotoba_pick.scene import build_scene, object_pos, reset_robot  # noqa: E402

GT = {"red cube": "red_cube", "green cube": "green_cube", "blue cube": "blue_cube", "yellow ball": "yellow_ball",
      "orange cylinder": "orange_can", "wooden box": "wooden_box", "white plate": "white_plate"}
CUBES = {"red cube": "red_cube", "green cube": "green_cube", "blue cube": "blue_cube"}


def yaw_err(est, gt):
    d = (est - gt) % (np.pi / 2)  # 立方体は 90° 対称
    return min(d, np.pi / 2 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_perception.json"))
    args = ap.parse_args()
    det = Detector()
    n_ok = n_tot = 0
    xy_errs, z_errs, yaw_errs, false_labels, times = [], [], [], [], []
    for seed in range(100, 100 + args.seeds):
        model, info = build_scene(seed=seed)
        data = mujoco.MjData(model)
        reset_robot(model, data)
        for _ in range(200):
            mujoco.mj_step(model, data)
        r = mujoco.Renderer(model, 720, 960)
        st = Perception(model, r, det).observe(data)
        r.close()
        times.append(st.seconds)
        found = {d.label: d for d in st.detections}
        false_labels += [lab for lab in found if lab not in GT]
        yaw_gt = {o.name: lay[2] for o, lay in zip(info["objects"], info["layout"])}
        for lab, body in GT.items():
            n_tot += 1
            d = found.get(lab)
            if d is None:
                continue
            p = object_pos(model, data, body)
            e = np.linalg.norm(d.center[:2] - p[:2])
            if e < 0.03:
                n_ok += 1
                xy_errs.append(e * 1000)
                if lab in CUBES:
                    z_errs.append(abs(d.top_z - 0.044) * 1000)
                    yaw_errs.append(np.rad2deg(yaw_err(d.yaw, yaw_gt[body])))
        print(f"seed {seed}: {len(found)} labels, {st.seconds:.1f}s")
    res = {
        "scenes": args.seeds, "label_accuracy": f"{n_ok}/{n_tot}",
        "xy_error_mm_mean": round(float(np.mean(xy_errs)), 1), "xy_error_mm_max": round(float(np.max(xy_errs)), 1),
        "cube_top_z_error_mm_mean": round(float(np.mean(z_errs)), 1),
        "cube_yaw_error_deg_mean": round(float(np.mean(yaw_errs)), 1),
        "cube_yaw_error_deg_max": round(float(np.max(yaw_errs)), 1),
        "false_labels": false_labels, "detector_seconds_mean": round(float(np.mean(times)), 1),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
