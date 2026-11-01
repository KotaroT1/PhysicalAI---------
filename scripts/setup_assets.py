"""Franka Panda のモデル (MuJoCo Menagerie, Apache-2.0) を third_party/ に取得する。

    python scripts/setup_assets.py

git のスパースチェックアウトで franka_emika_panda フォルダだけを取得する。
"""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST = ROOT / "third_party" / "mujoco_menagerie"
REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "cda5236f98383f6fbaa4820c6b18dba892cd087f"  # 動作確認したコミット


def run(*args, cwd=None):
    print("+", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def main():
    if (DEST / "franka_emika_panda" / "panda.xml").exists():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=DEST, capture_output=True, text=True).stdout.strip()
        if head and head != COMMIT:
            print(f"warning: existing assets are at {head}, expected {COMMIT}")
        print("already exists:", DEST / "franka_emika_panda")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", REPO, str(DEST))
    run("git", "sparse-checkout", "set", "franka_emika_panda", cwd=DEST)
    run("git", "checkout", COMMIT, cwd=DEST)
    print("done:", DEST / "franka_emika_panda")


if __name__ == "__main__":
    main()
