"""KotobaPick: 日本語の言語指示で動くテーブル片付けロボット (MuJoCo + ローカル AI)。"""
import os
import pathlib

# 学習済みモデルのキャッシュはプロジェクト内に閉じ込め、PC のユーザー環境を汚さない
_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
