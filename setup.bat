@echo off
echo HAIAのセットアップを開始します...

echo 1. 仮想環境(.venv)を作成中...
python -m venv .venv

echo 2. 必要なライブラリをインストール中...
call .venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo セットアップが完了しました！
echo ffmpeg.exe と pandoc.exe をフォルダに入れてから
echo launch.bat で起動してください。
pause