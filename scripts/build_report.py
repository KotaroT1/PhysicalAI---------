"""説明資料 (PDF) を docs/report.html から生成する (Microsoft Edge のヘッドレス印刷を使用)。

    python scripts/build_report.py --account ACCOUNT

Edge は一時プロファイル (--user-data-dir) で起動するため、普段のブラウザ設定には影響しない。
"""
import argparse
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EDGE_CANDIDATES = [
    pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    pathlib.Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", default="ACCOUNT", help="omnicampus アカウント名 (またはアカウントID)")
    args = ap.parse_args()
    edge = next((p for p in EDGE_CANDIDATES if p.exists()), None)
    if edge is None:
        raise SystemExit("Microsoft Edge が見つかりません。docs/report.html をブラウザで開いて PDF に印刷してください。")
    html = ROOT / "docs" / "report.html"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    pdf = out_dir / f"説明資料_PAI最終課題_{args.account}.pdf"
    with tempfile.TemporaryDirectory() as prof:
        tmp_pdf = pathlib.Path(prof) / "report.pdf"  # Edge に非 ASCII パスを渡さないよう一時ファイル経由
        subprocess.run([str(edge), "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                        f"--user-data-dir={prof}", "--no-pdf-header-footer", f"--print-to-pdf={tmp_pdf}",
                        html.resolve().as_uri()], check=True, timeout=180)
        shutil.copyfile(tmp_pdf, pdf)
    shutil.copyfile(pdf, ROOT / "docs" / "report.pdf")
    print("saved:", pdf)


if __name__ == "__main__":
    main()
