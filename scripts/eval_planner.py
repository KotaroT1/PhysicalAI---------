"""LLM タスク計画器の正解率評価 (シミュレーション不要)。

使い方:
    python scripts/eval_planner.py --model Qwen/Qwen2.5-1.5B-Instruct --mode choice
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from kotoba_pick.planner import LLMPlanner  # noqa: E402

OBJECTS = ["red cube", "green cube", "blue cube", "yellow ball", "orange cylinder"]
DESC = {"red cube": "right", "green cube": "front center", "blue cube": "left",
        "yellow ball": "center", "orange cylinder": "back center"}
PLACES = ["wooden box", "white plate"]
BOX, PLATE = "wooden box", "white plate"

# (指示, 期待される {物体: 行き先}) — 記載のない物体は動かさない
CASES = [
    ("赤いブロックを木箱に入れて", {"red cube": BOX}),
    ("黄色いボールをお皿にのせて", {"yellow ball": PLATE}),
    ("丸いものは全部お皿に、ブロックは全部木箱に片付けて",
     {"red cube": BOX, "green cube": BOX, "blue cube": BOX, "yellow ball": PLATE, "orange cylinder": PLATE}),
    ("緑のブロックを青いブロックの上に積んで", {"green cube": "blue cube"}),
    ("一番左にあるものをお皿にのせて", {"blue cube": PLATE}),
    ("机の上をきれいに片付けて",
     {"red cube": BOX, "green cube": BOX, "blue cube": BOX, "yellow ball": BOX, "orange cylinder": BOX}),
    ("黄色いボールと赤いブロックをお皿に置いてください", {"yellow ball": PLATE, "red cube": PLATE}),
    ("青と緑のブロックを箱にしまって", {"blue cube": BOX, "green cube": BOX}),
    ("オレンジの缶を木箱へ", {"orange cylinder": BOX}),
    ("ブロックを全部お皿に集めて", {"red cube": PLATE, "green cube": PLATE, "blue cube": PLATE}),
    # 開発後に追加した問題 (言い換え・存在しない物体)。存在しない物体は「何もしない」が正解
    ("缶とボールは皿にまとめておいて", {"orange cylinder": PLATE, "yellow ball": PLATE}),
    ("紫のブロックを木箱に入れて", {}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mode", default="decompose", choices=["decompose", "choice", "generate"])
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t = time.time()
    pl = LLMPlanner(args.model, mode=args.mode, think=args.think)
    print(f"model {args.model} ({args.mode}, think={args.think}) loaded in {time.time() - t:.1f}s")
    rows, n_ok = [], 0
    for ins, expect in CASES:
        p = pl.plan(ins, OBJECTS, PLACES, DESC)
        got = {s["pick"]: s["place"] for s in p.steps}  # 実行順は評価しない (順序はプログラムで決定)
        ok = got == expect
        n_ok += ok
        print(f"[{'OK' if ok else 'NG'}] {ins}  ({p.source}, {p.seconds:.1f}s)")
        if not ok:
            print(f"     expect {expect}\n     got    {got}")
        print(f"     think: {p.think}\n     reply: {p.reply}")
        rows.append({"instruction": ins, "ok": ok, "source": p.source, "seconds": round(p.seconds, 1),
                     "got": got, "expect": expect, "reply": p.reply, "think": p.think, "raw": p.raw})
    print(f"\naccuracy {n_ok}/{len(CASES)}, mean time {sum(r['seconds'] for r in rows) / len(rows):.1f}s")
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps({"model": args.model, "mode": args.mode, "think": args.think, "accuracy": n_ok / len(CASES),
                                                      "rows": rows}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
