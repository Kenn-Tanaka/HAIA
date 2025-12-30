# nuitka-project: --standalone
# nuitka-project: --enable-plugin=tk-inter
# nuitka-project: --windows-console-mode=disable
# nuitka-project: --lto=no
# nuitka-project: --plugin-enable=anti-bloat
# nuitka-project: --noinclude-pytest-mode=nofollow
# nuitka-project: --noinclude-setuptools-mode=nofollow
# nuitka-project: --include-package=keyring

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog, Menu, filedialog
import tkinter.font as tkfont
import threading
import requests
import os
import sys
import json
import queue
import keyring
import keyring.errors
import tempfile
import io
import shutil
import re
import wave
import pygame
import datetime
import argparse
import time
from pydub import AudioSegment

# ==========================================
# 公式SDKのインポート
# ==========================================
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False


# ==========================================
# 共通ユーティリティ (パス解決用)
# ==========================================
def get_exe_dir():
    """EXEファイルがあるディレクトリ（binフォルダ）を返す"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

# 【変更】起動時にディレクトリを固定するため、単純な絶対パス変換に変更
def resolve_path(filename):
    """設定ファイルのパスを解決する（CWD基準の絶対パスを返す）"""
    return os.path.abspath(filename)

# 【追加】起動時にカレントディレクトリを適切な場所に固定する関数
def setup_working_directory():
    """
    tasks.json をアンカーとしてルートディレクトリを特定し、
    カレントディレクトリ(CWD)をそこに固定する。
    """
    base_dir = get_exe_dir()
    
    # 探索候補: [現在のディレクトリ, 親ディレクトリ]
    # 親ディレクトリを優先探索対象に含めることで、binフォルダ構成に対応
    candidates = [base_dir, os.path.dirname(base_dir)]
    target_dir = base_dir # フォールバック（見つからない場合）

    # tasks.json がある場所を「正」とする
    # (定数はまだ定義前だが文字列リテラルとして扱うか、定数定義を上に移動するのが定石だが
    #  ここでは簡易的に文字列リテラル "tasks.json" を使用して判定する)
    anchor_file = "tasks.json"

    for d in candidates:
        if os.path.exists(os.path.join(d, anchor_file)):
            target_dir = d
            break
            
    # カレントディレクトリを変更
    os.chdir(target_dir)
    
    # import等でパス解決できるようsys.pathにも追加しておく
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    # デバッグ出力（引数解析前なのでprintを使用）
    # print(f"[Init] Working Directory set to: {target_dir}")

def find_external_tool(filename):
    """外部ツール(ffmpeg, pandoc)を探索する"""
    # CWDが固定されているので、まずはそこを探す
    cwd_path = os.path.abspath(filename)
    if os.path.exists(cwd_path): return cwd_path
    
    # binフォルダ（exeの場所）も探す
    bin_dir = get_exe_dir()
    path_in_bin = os.path.join(bin_dir, filename)
    if os.path.exists(path_in_bin): return path_in_bin
        
    return shutil.which(filename)

# ==========================================
# オプションライブラリの動的インポート
# ==========================================
# Pandoc
try:
    import pypandoc
    pandoc_path = find_external_tool("pandoc.exe")
    if pandoc_path and os.path.exists(pandoc_path):
        os.environ["PYPANDOC_PANDOC"] = pandoc_path
    try:
        pypandoc.get_pandoc_version()
        PANDOC_AVAILABLE = True
    except OSError:
        PANDOC_AVAILABLE = False
except ImportError:
    PANDOC_AVAILABLE = False

# PDF
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


# ==========================================
# 0. 定数・設定・ヘルパー
# ==========================================
APP_VERSION = "0.21"
SERVICE_NAME = "CloudLLM"
TASKS_FILENAME = "tasks.json"
SETTINGS_FILENAME = "settings.json"
MODELS_FILENAME = "models.json"
CASTING_CONFIG_FILENAME = "casting_config.json"
DEBUG_MODE = False
DEBUG_LOG_FILENAME = "debug.log"

DEFAULT_CASTING_CONFIG = {
    "system_instruction_template": "You are an expert Audio Casting Director. Analyze the text and assign voices.",
    "tts_direction_prompt": "Speak clearly and naturally.",
    "voices": {"Standard": {"names": ["Aoede", "Puck"], "description": "Standard voices."}},
    "fallback_voice": "Aoede"
}

DEFAULT_SETTINGS = {
    "window_width": 900, "window_height": 950, "font_size": 10,
    "temperature": 0.7, "request_timeout": 120,
    "last_model_index": 0, "last_task_index": 0,
    "custom_system_prompt": "あなたは優秀なAIアシスタントです。",
    # --- Pandoc用テンプレート設定 ---
    "reference_docx": "./templates/custom.docx",
    "reference_odt": "./templates/custom.odt",
    "reference_pptx": "./templates/custom.pptx"
}

DEFAULT_MODELS = {
  "Gemini 2.5 Flash Lite (Google)": "google/gemini-2.5-flash-lite",
  "Gemini 2.5 Flash (Google)": "google/gemini-2.5-flash",
  "Gemini 2.5 Pro (Google)": "google/gemini-2.5-pro",
  "GPT-5 nano (OpenAI)": "openai/gpt-5-nano-2025-08-07",
  "Claude 3.5 Sonnet (Anthropic)": "anthropic/claude-3-5-sonnet-latest",
  "GPT-OSS 120B (OpenRouter)": "openrouter/openai/gpt-oss-120b:free",
  "Qwen3 235B (OpenRouter)": "openrouter/qwen/qwen3-235b-a22b:free",
  "LM Studio (Local Server)": "lmstudio/local-model"
}

def log_debug(title, content):
    if not DEBUG_MODE: return
    try:
        path = resolve_path(DEBUG_LOG_FILENAME)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*20}\n[DEBUG] {timestamp} - {title}\n{'='*20}\n")
            f.write(str(content) + "\n\n")
    except Exception as e:
        print(f"[DEBUG] Logging failed: {e}")

class ConfigManager:
    def __init__(self):
        self.path = resolve_path(SETTINGS_FILENAME)
        self.config = DEFAULT_SETTINGS.copy()
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.config.update(data)
            except Exception as e: print(f"Settings Load Error: {e}")
        return self.config
    def save(self, data):
        try:
            with open(self.path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e: print(f"Settings Save Error: {e}")

# リソースローダー
def load_models_config():
    json_path = resolve_path(MODELS_FILENAME)
    if not os.path.exists(json_path): return DEFAULT_MODELS.copy(), False
    try:
        with open(json_path, 'r', encoding='utf-8') as f: return json.load(f), True
    except: return DEFAULT_MODELS.copy(), False

def load_system_prompts():
    json_path = resolve_path(TASKS_FILENAME)
    default = {"デフォルト": "あなたは有能なAIアシスタントです。"}
    if not os.path.exists(json_path): return default, False
    try:
        with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        resolved = {}
        # CWDが固定されているので、絶対パス変換が正確に行われる
        base_dir = os.path.dirname(os.path.abspath(json_path))
        for k, v in data.items():
            if isinstance(v, str) and v.startswith("./"):
                fp = os.path.normpath(os.path.join(base_dir, v))
                if os.path.exists(fp):
                    with open(fp, 'r', encoding='utf-8') as pf: resolved[k] = pf.read()
                else: resolved[k] = f"Error: File not found {v}"
            else: resolved[k] = v
        return resolved, True
    except: return default, False

def load_casting_config():
    json_path = resolve_path(CASTING_CONFIG_FILENAME)
    if not os.path.exists(json_path): return DEFAULT_CASTING_CONFIG.copy()
    try:
        with open(json_path, 'r', encoding='utf-8') as f: return json.load(f)
    except: return DEFAULT_CASTING_CONFIG.copy()

class KeyManager:
    @staticmethod
    def _make_username(vendor): return f"{vendor}|research|default"
    @classmethod
    def save_key(cls, v, k): 
        try: return keyring.set_password(SERVICE_NAME, cls._make_username(v), k) is None or True
        except: return False
    @classmethod
    def get_key(cls, v):
        try: return keyring.get_password(SERVICE_NAME, cls._make_username(v))
        except: return None
    @classmethod
    def delete_key(cls, v):
        try: keyring.delete_password(SERVICE_NAME, cls._make_username(v)); return True
        except: return False

# ==========================================
# 1. LLMハンドラー
# ==========================================
class LLMHandler:
    PROVIDER_KEY_MAP = {"google": "gemini"}
    
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url

    def get_installed_ollama_models(self):
        try:
            res = requests.get(f"{self.ollama_url}/api/tags", timeout=1)
            if res.status_code == 200: return [m['name'] for m in res.json().get('models', [])]
        except: return []
        return []

    def stream_response(self, full_id, system_prompt, user_text, temperature=0.7, timeout=120):
        provider, model_name = full_id.split("/", 1)
        req_timeout = float(timeout) if timeout > 0 else None
        
        # --- Google Gemini ---
        if provider == "google":
            if not GEMINI_SDK_AVAILABLE: raise ImportError("google-genaiライブラリが必要です")
            api_key = KeyManager.get_key("gemini") or os.environ.get("GEMINI_API_KEY")
            if not api_key: raise ValueError("Gemini APIキー未設定")
            
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_prompt
            )
            response = client.models.generate_content_stream(
                model=model_name,
                contents=user_text,
                config=config
            )
            for chunk in response:
                if chunk.text: yield chunk.text

        # --- OpenAI / LM Studio / OpenRouter ---
        elif provider in ["openai", "openrouter", "lmstudio"]:
            if not openai: raise ImportError("openaiライブラリが必要です")
            api_key = None
            base_url = None
            
            if provider == "openai":
                api_key = KeyManager.get_key("openai") or os.environ.get("OPENAI_API_KEY")
                if not api_key: raise ValueError("OpenAI APIキー未設定")
            elif provider == "openrouter":
                api_key = KeyManager.get_key("openrouter") or os.environ.get("OPENROUTER_API_KEY")
                base_url = "https://openrouter.ai/api/v1"
                if not api_key: raise ValueError("OpenRouter APIキー未設定")
            elif provider == "lmstudio":
                api_key = "lm-studio"
                base_url = "http://localhost:1234/v1"

            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=req_timeout)
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content: yield content

        # --- Anthropic (Claude) ---
        elif provider == "anthropic":
            if not anthropic: raise ImportError("anthropicライブラリが必要です")
            api_key = KeyManager.get_key("anthropic") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("Anthropic APIキー未設定")

            client = anthropic.Anthropic(api_key=api_key, timeout=req_timeout)
            with client.messages.stream(
                max_tokens=4096,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_text}],
                model=model_name,
            ) as stream:
                for text in stream.text_stream: yield text

        # --- Ollama (requests直接) ---
        elif provider == "ollama":
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                "options": {"temperature": temperature},
                "stream": True
            }
            with requests.post(f"{self.ollama_url}/api/chat", json=payload, stream=True, timeout=req_timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        try:
                            body = json.loads(line)
                            content = body.get("message", {}).get("content", "")
                            if content: yield content
                            if body.get("done", False): break
                        except: pass
        else:
            raise ValueError(f"未対応のプロバイダ: {provider}")

    def generate_text_oneshot(self, full_id, system_prompt, user_text, temperature=0.0):
        full_text = ""
        for chunk in self.stream_response(full_id, system_prompt, user_text, temperature=temperature):
            full_text += chunk
        return full_text

# ==========================================
# 2. 音声合成ハンドラー
# ==========================================
class AudioHandler:
    def __init__(self):
        pygame.mixer.init()
        self.gemini_client = None
        self.ffmpeg_path = find_external_tool("ffmpeg.exe")
        self.ffmpeg_available = self.ffmpeg_path is not None
        if self.ffmpeg_available:
            AudioSegment.converter = self.ffmpeg_path
        else:
            print("[Warning] FFmpegが見つかりません。")

    def reset_client(self):
        self.gemini_client = None

    def _get_client(self):
        if self.gemini_client: return self.gemini_client
        if not GEMINI_SDK_AVAILABLE: raise ImportError("google-genai ライブラリが必要です。")
        api_key = KeyManager.get_key("gemini") or os.environ.get("GEMINI_API_KEY")
        if not api_key: raise ValueError("Gemini API Keyが設定されていません。")
        self.gemini_client = genai.Client(api_key=api_key)
        return self.gemini_client

    def is_ready(self):
        return (KeyManager.get_key("gemini") or os.environ.get("GEMINI_API_KEY")) is not None

    def stop_playback(self):
        if pygame.mixer.get_init(): pygame.mixer.music.stop()

    def _raw_pcm_to_wav(self, pcm_data):
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1) 
                wav_file.setsampwidth(2) 
                wav_file.setframerate(24000) 
                wav_file.writeframes(pcm_data)
            return wav_buffer.getvalue()

    def _gen_gemini_dialogue(self, text, voice_map, config_obj, is_script=True):
        client = self._get_client()
        processed_lines = []
        line_pattern = re.compile(r"^(.+?)(\s*\(.*?\))?\s*[:：]\s*(.+)$")

        if is_script:
            for line in text.split('\n'):
                line = line.strip()
                if not line: continue
                line = re.sub(r"^(.+?)[「『](.*?)[」』]$", r"\1: \2", line)
                match = line_pattern.match(line)
                if match:
                    processed_lines.append(f"{match.group(1).strip()}: {match.group(3).strip()}")
                else:
                    processed_lines.append(line)
        else:
            primary_speaker = list(voice_map.keys())[0] if voice_map else "Narrator"
            for line in text.split('\n'):
                if line.strip(): processed_lines.append(f"{primary_speaker}: {line.strip()}")

        final_script = "\n".join(processed_lines)
        active_speakers = list(voice_map.keys())[:2] 
        
        speaker_configs = []
        used_voices = set()
        default_voice = config_obj.get("fallback_voice", "Aoede")

        for name in active_speakers:
            voice = voice_map.get(name, default_voice)
            speaker_configs.append({"speaker": name, "voice_config": {"prebuilt_voice_config": {"voice_name": voice}}})
            used_voices.add(voice)
            
        while len(speaker_configs) < 2:
            dummy_name = f"Silent_{len(speaker_configs)}"
            dummy_voice = "Puck" if "Puck" not in used_voices else "Aoede"
            speaker_configs.append({"speaker": dummy_name, "voice_config": {"prebuilt_voice_config": {"voice_name": dummy_voice}}})

        direction = config_obj.get("tts_direction_prompt", "Speak clearly.")
        prompt = f"""Generate audio.
        # Direction: {direction}
        # Voice Assignment: {voice_map}
        # Content: {final_script}"""

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=prompt,
            config={
                "response_modalities": ["AUDIO"],
                "speech_config": {"multi_speaker_voice_config": {"speaker_voice_configs": speaker_configs}}
            }
        )
        
        if response.candidates and response.candidates[0].content.parts:
            return self._raw_pcm_to_wav(response.candidates[0].content.parts[0].inline_data.data)
        else:
            raise Exception("Geminiから音声データが返されませんでした。")

    def save_to_file(self, text, voice_map, is_script, filepath, config_obj, status_callback=None):
        if not self.is_ready():
            if status_callback: status_callback("エラー: API Key未設定")
            return
        threading.Thread(target=self._run_save, args=(text, voice_map, is_script, filepath, config_obj, status_callback), daemon=True).start()

    def generate_and_play(self, text, voice_map, is_script, config_obj, status_callback=None):
        if not self.is_ready():
            if status_callback: status_callback("エラー: API Key未設定")
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_name = tmp.name
        threading.Thread(target=self._run_save, args=(text, voice_map, is_script, tmp_name, config_obj, status_callback, True), daemon=True).start()

    def _run_save(self, text, voice_map, is_script, filepath, config_obj, status_callback, auto_play=False):
        try:
            if status_callback: status_callback("音声生成中 (Gemini TTS)...")
            wav_data = self._gen_gemini_dialogue(text, voice_map, config_obj, is_script=is_script)
            
            if not self.ffmpeg_available and not filepath.lower().endswith(".wav"):
                base = os.path.splitext(filepath)[0]
                filepath = base + ".wav"
                if status_callback: status_callback("※FFmpeg未検出のためWAV保存になります")

            if status_callback: status_callback("書き出し中...")
            
            if filepath.lower().endswith(".wav"):
                with open(filepath, "wb") as f: f.write(wav_data)
            else:
                segment = AudioSegment.from_wav(io.BytesIO(wav_data))
                segment.export(filepath, format="mp3", bitrate="192k")

            if auto_play:
                if status_callback: status_callback("再生中...")
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy(): pygame.time.wait(100)
                pygame.mixer.music.unload()
                time.sleep(0.2)
                try: os.remove(filepath)
                except: pass
                if status_callback: status_callback("再生完了")
            else:
                if status_callback: status_callback(f"保存完了: {os.path.basename(filepath)}")

        except Exception as e:
            msg = f"Audio Error: {str(e)}"
            print(msg)
            if status_callback: status_callback(f"エラー: {str(e)[:40]}...")

# ==========================================
# 3. GUIアプリケーション
# ==========================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"HAIA Ver.{APP_VERSION}" + (" [DEBUG]" if DEBUG_MODE else ""))
        
        self.config_manager = ConfigManager()
        self.settings = self.config_manager.load()
        
        # 【変更】初期化時のロード処理を削除し、空で初期化（重複防止）
        self.cloud_models = {} 
        self.system_prompts = {}
        
        self.casting_config = load_casting_config()
        
        self.geometry(f"{self.settings.get('window_width', 900)}x{self.settings.get('window_height', 950)}")
        
        self.handler = LLMHandler()
        self.audio = AudioHandler()
        self.dynamic_model_map = {} 
        self.msg_queue = queue.Queue()
        self.is_running = False
        self.base_font_size = self.settings.get("font_size", 10)
        self.custom_font = tkfont.Font(family="Meiryo UI", size=self.base_font_size)
        
        self._create_menu()
        self._init_ui() # UI構築メソッド（タブ構成に変更）
        
        # モデルリスト取得時にタスクリストも更新されるため、ここで一度だけ走らせる
        self.after(500, lambda: self._refresh_model_list(silent=True))
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(1000, self._check_gemini_key)

    def on_close(self):
        self.audio.stop_playback()
        self.settings.update({
            "window_width": self.winfo_width(),
            "window_height": self.winfo_height(),
            "font_size": self.base_font_size,
            "temperature": self.scale_temp.get(),
            "last_model_index": self.combo_model.current(),
            "last_task_index": self.combo_task.current(),
            "custom_system_prompt": self.text_system_prompt.get("1.0", tk.END).strip()
        })
        self.config_manager.save(self.settings)
        self.destroy()

    def _create_menu(self):
        menubar = Menu(self)
        self.config(menu=menubar)
        s_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="設定", menu=s_menu)
        # Ver 0.20 (追加) データフォルダを開く
        s_menu.add_command(label="データフォルダを開く...", command=lambda: self._open_folder())
        s_menu.add_separator()
        s_menu.add_command(label="設定再読込", command=self._reload_configs)
        s_menu.add_separator()
        for v in ["gemini", "openai", "anthropic", "openrouter"]:
            s_menu.add_command(label=f"{v} Key 登録...", command=lambda V=v: self._ask_key(V))
        s_menu.add_separator()
        for v in ["gemini", "openai", "anthropic", "openrouter"]:
            s_menu.add_command(label=f"{v} Key 削除", command=lambda V=v: self._delete_key(V))
        v_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="表示", menu=v_menu)
        v_menu.add_command(label="文字拡大", command=lambda: self._change_font_size(1))
        v_menu.add_command(label="文字縮小", command=lambda: self._change_font_size(-1))

    # ==========================================
    # UI構築 (Ver0.19 タブ化対応)
    # ==========================================
    def _init_ui(self):
        # Notebook (タブコンテナ) の作成
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self.tab_main = ttk.Frame(self.notebook)
        self.tab_editor = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_main, text="テキスト生成") 
        self.notebook.add(self.tab_editor, text="プロンプト管理")

        # 各タブの中身を初期化
        self._init_main_tab()
        self._init_editor_tab()
        
        # イベントバインド
        self.bind('<Control-Return>', lambda e: self.on_run())
        # タブ切り替え時にリストをリロードする
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _init_main_tab(self):
        # --- 既存のメイン画面UI (parentを self.tab_main に変更) ---
        control_frame = ttk.LabelFrame(self.tab_main, text="AI設定", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        f_model = ttk.Frame(control_frame)
        f_model.pack(fill="x", pady=2)
        ttk.Label(f_model, text="モデル:").pack(side="left")
        self.combo_model = ttk.Combobox(f_model, state="readonly", width=40)
        self.combo_model.pack(side="left", padx=5)
        self.btn_model_refresh = ttk.Button(f_model, text="更新", width=4, command=lambda: self._refresh_model_list(False))
        self.btn_model_refresh.pack(side="left")

        f_prompt_ctrl = ttk.Frame(control_frame)
        f_prompt_ctrl.pack(fill="x", pady=(10, 2))
        ttk.Label(f_prompt_ctrl, text="プリセット:").pack(side="left")
        
        # 【変更】初期化時点でデータが空の場合に備える
        initial_tasks = sorted(list(self.system_prompts.keys())) if self.system_prompts else []
        self.combo_task = ttk.Combobox(f_prompt_ctrl, values=initial_tasks, state="readonly", width=30)
        self.combo_task.pack(side="left", padx=5)
        self.combo_task.bind("<<ComboboxSelected>>", self._on_preset_selected)
        
        ttk.Label(control_frame, text="システムプロンプト (編集可 - 一時的):").pack(anchor="w", pady=(5, 0))
        self.text_system_prompt = scrolledtext.ScrolledText(control_frame, height=5, font=("Meiryo UI", 9))
        self.text_system_prompt.pack(fill="both", expand=True, padx=5, pady=5)
        self.text_system_prompt.insert("1.0", self.settings.get("custom_system_prompt", ""))
        self._bind_context_menu(self.text_system_prompt)

        f_temp = ttk.Frame(control_frame)
        f_temp.pack(fill="x", pady=2)
        ttk.Label(f_temp, text="創造性 (Temp):").pack(side="left")
        self.scale_temp = tk.Scale(f_temp, from_=0.0, to=1.0, resolution=0.1, orient="horizontal", length=200)
        self.scale_temp.set(self.settings.get("temperature", 0.7))
        self.scale_temp.pack(side="left", padx=10)

        input_container = ttk.LabelFrame(self.tab_main, text="入力テキスト", padding=(10, 5, 10, 10))
        input_container.pack(fill="both", expand=True, padx=10, pady=5)
        tb_in = ttk.Frame(input_container)
        tb_in.pack(fill="x")
        ttk.Button(tb_in, text="📂 読込", command=self._load_file_to_input).pack(side="left")
        ttk.Button(tb_in, text="🗑️ クリア", command=lambda: self.text_input.delete("1.0", tk.END)).pack(side="right")
        self.text_input = scrolledtext.ScrolledText(input_container, height=6, font=self.custom_font)
        self.text_input.pack(fill="both", expand=True)
        self._bind_context_menu(self.text_input)
        self.text_input.bind("<Control-MouseWheel>", self._on_zoom)

        self.btn_run = ttk.Button(self.tab_main, text="▶ テキスト生成 (Ctrl+Enter)", command=self.on_run)
        self.btn_run.pack(pady=5, ipadx=40, ipady=5)

        output_container = ttk.LabelFrame(self.tab_main, text="AIの回答 / 音声操作", padding=(10, 5, 10, 10))
        output_container.pack(fill="both", expand=True, padx=10, pady=5)
        tb_out = ttk.Frame(output_container)
        tb_out.pack(fill="x")
        
        self.lbl_engine_status = ttk.Label(tb_out, text="TTS: 準備中", foreground="gray")
        self.lbl_engine_status.pack(side="left")
        ttk.Button(tb_out, text="💾 保存", command=self._save_output).pack(side="left", padx=5)
        ttk.Button(tb_out, text="📋 コピー", command=self._copy_output).pack(side="left")
        ttk.Separator(tb_out, orient="vertical").pack(side="left", padx=5, fill="y")
        ttk.Button(tb_out, text="🔊 自動配役＆再生", command=self._play_audio_with_casting).pack(side="left")
        ttk.Button(tb_out, text="⏹", width=3, command=self._stop_audio).pack(side="left")
        ttk.Button(tb_out, text="🎵 自動配役＆保存", command=self._save_audio_with_casting).pack(side="left", padx=5)
        
        ttk.Button(tb_out, text="⬇️ 入力→出力", command=self._copy_input_to_output).pack(side="right", padx=2)
        ttk.Button(tb_out, text="🔁 入替", command=self._swap_input_output).pack(side="right", padx=2)
        ttk.Button(tb_out, text="⬆️ 転記", command=self._transfer_output).pack(side="right", padx=2)
        
        text_frame = ttk.Frame(output_container)
        text_frame.pack(fill="both", expand=True)
        h_scroll = ttk.Scrollbar(text_frame, orient="horizontal")
        v_scroll = ttk.Scrollbar(text_frame, orient="vertical")
        self.text_output = tk.Text(
            text_frame, height=10, state="disabled", bg="#f8f8f8", font=self.custom_font,
            wrap=tk.NONE, xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set
        )
        h_scroll.config(command=self.text_output.xview)
        v_scroll.config(command=self.text_output.yview)
        self.text_output.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        self._bind_context_menu(self.text_output)
        self.text_output.bind("<Control-MouseWheel>", self._on_zoom)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.tab_main, textvariable=self.status_var, relief="sunken").pack(side="bottom", fill="x")

    def _init_editor_tab(self):
        self.current_editing_key = None  # リネーム判定用

        # --- プロンプト管理用の新規タブ ---
        paned = ttk.PanedWindow(self.tab_editor, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 左側: タスクリスト
        left_frame = ttk.Frame(paned, width=200)
        paned.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="プロンプト一覧").pack(anchor="w")
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill="both", expand=True)
        self.lb_tasks = tk.Listbox(list_frame, exportselection=False)
        self.lb_tasks.pack(side="left", fill="both", expand=True)
        sb_tasks = ttk.Scrollbar(list_frame, orient="vertical", command=self.lb_tasks.yview)
        sb_tasks.pack(side="right", fill="y")
        self.lb_tasks.config(yscrollcommand=sb_tasks.set)
        self.lb_tasks.bind("<<ListboxSelect>>", self._on_task_select_editor)
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="＋ 新規追加", command=self._add_new_task).pack(side="left", expand=True, fill="x")
        ttk.Button(btn_frame, text="－ 削除", command=self._delete_task).pack(side="left", expand=True, fill="x")

        # 右側: 編集エリア
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        
        f_name = ttk.Frame(right_frame)
        f_name.pack(fill="x", pady=5)
        ttk.Label(f_name, text="タスク名(キー):").pack(side="left")
        self.ent_task_key = ttk.Entry(f_name)
        self.ent_task_key.pack(side="left", fill="x", expand=True, padx=5)
        self._bind_context_menu(self.ent_task_key) # ★右クリックメニュー追加
        
        f_path = ttk.Frame(right_frame)
        f_path.pack(fill="x", pady=5)
        ttk.Label(f_path, text="保存先ファイル:").pack(side="left")
        self.ent_task_path = ttk.Entry(f_path)
        self.ent_task_path.pack(side="left", fill="x", expand=True, padx=5)
        self._bind_context_menu(self.ent_task_path) # ★右クリックメニュー追加

        # 「...」ボタンでファイル選択ダイアログを開く
        ttk.Button(f_path, text="...", width=3, command=self._browse_task_path).pack(side="left", padx=2)
        ttk.Button(f_path, text="📂", width=3, command=self._open_current_task_folder).pack(side="left", padx=2)
        ttk.Label(f_path, text="(空白=JSON直埋込)").pack(side="left")

        ttk.Label(right_frame, text="プロンプト内容:").pack(anchor="w")
        self.txt_editor_content = scrolledtext.ScrolledText(right_frame, font=self.custom_font)
        self.txt_editor_content.pack(fill="both", expand=True)
        self._bind_context_menu(self.txt_editor_content) # ★右クリックメニュー追加
        
        ttk.Button(right_frame, text="💾 変更を保存", command=self._save_editor_changes).pack(anchor="e", pady=5)
        
        self.raw_tasks_data = {} 
        self._reload_editor_list()

    # ==========================================
    # 内部ロジック & イベント
    # ==========================================

    def _bind_context_menu(self, w):
        m = Menu(w, tearoff=0)
        m.add_command(label="Cut", command=lambda: w.event_generate("<<Cut>>"))
        m.add_command(label="Copy", command=lambda: w.event_generate("<<Copy>>"))
        m.add_command(label="Paste", command=lambda: w.event_generate("<<Paste>>"))
        m.add_separator()
        
        def select_all():
            w.focus_set()
            # Entry (tk or ttk)の場合は select_range
            if isinstance(w, (tk.Entry, ttk.Entry)):
                w.select_range(0, tk.END)
            else:
                # Text / ScrolledText の場合は tag_add
                try:
                    w.tag_add("sel", "1.0", "end")
                except:
                    pass 

        m.add_command(label="Select All", command=select_all)
        w.bind("<Button-3>", lambda e: m.tk_popup(e.x_root, e.y_root))

    def _on_preset_selected(self, event):
        task = self.combo_task.get()
        self.text_system_prompt.delete("1.0", tk.END)
        self.text_system_prompt.insert(tk.END, self.system_prompts.get(task, ""))

    def _check_gemini_key(self):
        # ★修正: 環境変数もチェック
        key = KeyManager.get_key("gemini") or os.environ.get("GEMINI_API_KEY")
        if key: self.lbl_engine_status.config(text="TTS: Gemini (OK)", foreground="green")
        else: self.lbl_engine_status.config(text="TTS: Gemini (Key未設定)", foreground="red")
    
    def _ask_key(self, v):
        k = simpledialog.askstring(f"{v} Key", "API Key (Ctrl+Vで貼付):", show='*', parent=self)
        if k: 
            KeyManager.save_key(v, k)
            if v == "gemini":
                self.audio.reset_client()
                self._check_gemini_key()
    
    def _delete_key(self, v):
        if messagebox.askyesno("Del", f"Delete {v} Key?"): KeyManager.delete_key(v); 
        if v == "gemini": self._check_gemini_key()

    def _refresh_model_list(self, silent=False):
        if hasattr(self, 'btn_model_refresh'):
            self.btn_model_refresh.config(state="disabled")
        if not silent:
            self.status_var.set("モデルリストを取得中...")
        thread = threading.Thread(target=self._thread_fetch_models, args=(silent,), daemon=True)
        thread.start()

    def _thread_fetch_models(self, silent):
        # 起動後のデータリロードをここで行う
        self.cloud_models, _ = load_models_config()
        self.system_prompts, _ = load_system_prompts() # タスクリストも更新
        
        current_map = self.cloud_models.copy()
        ollama_models = self.handler.get_installed_ollama_models()
        for m in ollama_models:
            current_map[f"[Local] {m}"] = f"ollama/{m}"
        self.after(0, self._update_model_ui, current_map, silent)

    def _update_model_ui(self, model_map, silent):
        self.dynamic_model_map = model_map
        self.combo_model['values'] = list(self.dynamic_model_map.keys())
        last_idx = self.settings.get("last_model_index", 0)
        if self.combo_model['values']:
            if last_idx < len(self.combo_model['values']): self.combo_model.current(last_idx)
            else: self.combo_model.current(0)
        
        # タスクリスト更新（ここでもソート）
        self.combo_task['values'] = sorted(list(self.system_prompts.keys()))
        last_task_idx = self.settings.get("last_task_index", -1)
        if 0 <= last_task_idx < len(self.combo_task['values']):
            self.combo_task.current(last_task_idx)

        if hasattr(self, 'btn_model_refresh'): self.btn_model_refresh.config(state="normal")
        if not silent: self.status_var.set("モデルリスト更新完了")

    def on_run(self):
        if self.is_running:
            self.is_running = False
            self.status_var.set("中断しました")
            self.btn_run.config(text="▶ テキスト生成 (Ctrl+Enter)")
            return
        input_text = self.text_input.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("警告", "入力テキストが空です")
            return
        display_name = self.combo_model.get()
        if not display_name: return
        full_id = self.dynamic_model_map.get(display_name, "")
        
        sys_prompt = self.text_system_prompt.get("1.0", tk.END).strip()
        temperature = self.scale_temp.get()

        self.is_running = True
        self.btn_run.config(text="■ 中断")
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        self.status_var.set(f"生成中... (Temp: {temperature})")
        
        # 生成時は自動的にメインタブを選択状態にする
        self.notebook.select(self.tab_main)

        thread = threading.Thread(target=self._run_thread, args=(full_id, sys_prompt, input_text, temperature))
        thread.start()
        self.after(100, self._process_queue)

    def _run_thread(self, full_id, sys_prompt, text, temp):
        try:
            for chunk in self.handler.stream_response(full_id, sys_prompt, text, temp):
                if not self.is_running: return
                self.msg_queue.put(("data", chunk))
            if self.is_running: self.msg_queue.put(("finish", None))
        except Exception as e:
            if self.is_running: self.msg_queue.put(("error", str(e)))

    def _process_queue(self):
        try:
            while True:
                m_type, content = self.msg_queue.get_nowait()
                if m_type == "data":
                    self.text_output.insert(tk.END, content)
                    self.text_output.see(tk.END)
                elif m_type == "finish": self._end_proc("完了"); return
                elif m_type == "error":
                    self.text_output.insert(tk.END, f"\n[Error] {content}")
                    self._end_proc("エラー発生")
                    return
        except queue.Empty:
            if self.is_running: self.after(100, self._process_queue)
            else: self._end_proc("中断")

    def _end_proc(self, msg):
        self.is_running = False
        self.btn_run.config(text="▶ テキスト生成 (Ctrl+Enter)")
        self.text_output.config(state="disabled")
        self.status_var.set(msg)

    # --- ヘルパー: 自動ファイル名生成 (Ver0.19) ---
    def _make_auto_filename(self, ext):
        base_name = self.combo_task.get()
        if not base_name: base_name = "Output"
        # ファイル名禁止文字を置換
        safe_name = re.sub(r'[\\/:*?"<>|]+', '_', base_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_name}_{timestamp}{ext}"

    # --- 配役・音声機能 ---
    def _extract_json_string(self, text):
        text = text.strip()
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        idx = text.find('{')
        if idx == -1: return None
        text = text[idx:]
        balance = 0
        for i, char in enumerate(text):
            if char == '{': balance += 1
            elif char == '}':
                balance -= 1
                if balance == 0: return text[:i+1]
        return None

    def _perform_casting(self, text):
        self.status_var.set("配役・構成を分析中...")
        display_name = self.combo_model.get()
        full_id = self.dynamic_model_map.get(display_name, "")
        
        valid_voices = set()
        category_map = {} 
        voice_list_str = ""
        for cat, info in self.casting_config.get("voices", {}).items():
            names_in_cat = info.get("names", [])
            for name in names_in_cat:
                valid_voices.add(name.lower())
            if names_in_cat: category_map[cat] = names_in_cat[0]
            voice_list_str += f"- [{cat}] Names: {', '.join(names_in_cat)} | {info.get('description','')}\n"

        system_instruction = f"""
        {self.casting_config.get("system_instruction_template")}
        # Available Voices
        {voice_list_str}
        Output strictly in JSON format: {{ "casting": [ {{"character_name": "...", "assigned_voice": "..."}} ], "is_script": true/false }}
        """

        try:
            result_text = self.handler.generate_text_oneshot(full_id, system_instruction, text, temperature=0.0)
            json_str = self._extract_json_string(result_text)
            if not json_str: raise ValueError("JSONが見つかりませんでした")
            data = json.loads(json_str)
            voice_map = {}
            fallback = self.casting_config.get("fallback_voice", "Aoede")
            for item in data.get("casting", []):
                char = item["character_name"]
                raw_voice = item["assigned_voice"].strip()
                if raw_voice.lower() in valid_voices: voice_map[char] = raw_voice
                elif raw_voice in category_map: voice_map[char] = category_map[raw_voice]
                else: voice_map[char] = fallback
            return voice_map, data.get("is_script", False)
        except Exception as e:
            print(f"Casting Error: {e}")
            messagebox.showerror("配役エラー", f"配役の決定に失敗しました。\n{e}")
            return None, False

    def _play_audio_with_casting(self):
        text = self.text_output.get("1.0", tk.END).strip()
        if not text: return
        threading.Thread(target=self._run_cast_and_play, args=(text,), daemon=True).start()

    def _run_cast_and_play(self, text):
        voice_map, is_script = self._perform_casting(text)
        if voice_map:
            self.audio.generate_and_play(text, voice_map, is_script, self.casting_config, status_callback=lambda m: self.status_var.set(m))

    def _save_audio_with_casting(self):
        text = self.text_output.get("1.0", tk.END).strip()
        if not text: return
        if getattr(self.audio, "ffmpeg_available", False):
            ft = [("MP3 Audio", "*.mp3"), ("WAV Audio", "*.wav")]
            d_ext = ".mp3"
        else:
            ft = [("WAV Audio", "*.wav")]
            d_ext = ".wav"

        default_filename = self._make_auto_filename(d_ext) # 自動ファイル名

        filename = filedialog.asksaveasfilename(
            defaultextension=d_ext, filetypes=ft, 
            initialfile=default_filename,
            title="音声を保存"
        )
        if not filename: return
        threading.Thread(target=self._run_cast_and_save, args=(text, filename), daemon=True).start()

    def _run_cast_and_save(self, text, filename):
        voice_map, is_script = self._perform_casting(text)
        if voice_map:
            def cb(m):
                self.status_var.set(m)
                if "完了" in m or "エラー" in m: messagebox.showinfo("Info", m)
            self.audio.save_to_file(text, voice_map, is_script, filename, self.casting_config, status_callback=cb)

    def _stop_audio(self):
        self.audio.stop_playback()
        self.status_var.set("再生停止")
    
    # --- 設定管理 ---
    def _reload_configs(self):
        self._refresh_model_list()
        self.system_prompts, _ = load_system_prompts()
        self.casting_config = load_casting_config()
        self.combo_task['values'] = sorted(list(self.system_prompts.keys()))
        # エディタ画面もリロード
        self._reload_editor_list()
        messagebox.showinfo("更新", "設定を再読み込みしました")

    def _on_zoom(self, event):
        d = 1 if event.delta > 0 else -1
        new_s = self.base_font_size + d
        if 6 <= new_s <= 40:
            self.base_font_size = new_s
            self.custom_font.configure(size=new_s)
        return "break"

    def _change_font_size(self, d):
        new_s = self.base_font_size + d
        if 6 <= new_s <= 40:
            self.base_font_size = new_s
            self.custom_font.configure(size=new_s)

    def _load_file_to_input(self):
        ft = [("All Supported", "*.txt *.md *.pdf *.docx *.odt *.epub *.html"), ("Text", "*.txt *.md")]
        filepath = filedialog.askopenfilename(filetypes=ft)
        if not filepath: return
        try:
            content = ""
            ext = os.path.splitext(filepath)[1].lower()
            if PYPDF_AVAILABLE and ext == ".pdf":
                reader = PdfReader(filepath)
                content = "\n\n".join([p.extract_text() or "" for p in reader.pages])
            elif PANDOC_AVAILABLE and ext in [".docx", ".odt", ".epub", ".html"]:
                content = pypandoc.convert_file(filepath, 'markdown')
            else:
                with open(filepath, "r", encoding="utf-8") as f: content = f.read()
            self.text_input.delete("1.0", tk.END)
            self.text_input.insert(tk.END, content)
            self.status_var.set(f"読込完了: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_output(self):
        if PANDOC_AVAILABLE:
            ft = [("Text File", "*.txt"), ("Markdown", "*.md"), ("Word Document", "*.docx"),
                  ("PowerPoint", "*.pptx"), ("OpenDocument Text", "*.odt"),
                  ("HTML File", "*.html"), ("EPUB Book", "*.epub")]
            d_ext = ".txt"
        else:
            ft = [("Text File", "*.txt")]
            d_ext = ".txt"

        default_filename = self._make_auto_filename(d_ext) # 自動ファイル名

        fp = filedialog.asksaveasfilename(
            defaultextension=d_ext, filetypes=ft,
            initialfile=default_filename, 
            title="保存先を選択"
        )
        if not fp: return

        try:
            content = self.text_output.get("1.0", tk.END)
            ext = os.path.splitext(fp)[1].lower()
            pandoc_exts = [".docx", ".odt", ".epub", ".html", ".pptx"]

            if PANDOC_AVAILABLE and ext in pandoc_exts:
                extra_args = ['--standalone']
                ref_map = {".docx": "reference_docx", ".odt": "reference_odt", ".pptx": "reference_pptx"}
                if ext in ref_map:
                    ref_path_setting = self.settings.get(ref_map[ext], "")
                    if ref_path_setting:
                        abs_ref_path = resolve_path(ref_path_setting)
                        if os.path.exists(abs_ref_path):
                            extra_args.append(f'--reference-doc={abs_ref_path}')
                            print(f"[Info] Applied reference doc: {abs_ref_path}")
                        else:
                            print(f"[Warning] Reference doc not found: {abs_ref_path}")
                output_format = ext.replace('.', '')
                pypandoc.convert_text(content, to=output_format, format='markdown', outputfile=fp, extra_args=extra_args)
            else:
                with open(fp, "w", encoding="utf-8") as f: f.write(content)
            self.status_var.set(f"保存しました: {os.path.basename(fp)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Save Error: {e}")

    def _copy_output(self):
        self.clipboard_clear()
        self.clipboard_append(self.text_output.get("1.0", tk.END))
        self.status_var.set("コピーしました")

    def _transfer_output(self):
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert(tk.END, self.text_output.get("1.0", tk.END))

    def _copy_input_to_output(self):
        src = self.text_input.get("1.0", tk.END)
        self._set_output_text(src)
        self.status_var.set("入力テキストをAI出力欄へコピーしました")

    def _swap_input_output(self):
        in_text = self.text_input.get("1.0", tk.END)
        out_text = self.text_output.get("1.0", tk.END)
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert(tk.END, out_text)
        self._set_output_text(in_text)
        self.status_var.set("入力/出力テキストを入れ替えました")

    def _set_output_text(self, text: str):
        prev_state = str(self.text_output.cget("state"))
        try:
            if prev_state != "normal": self.text_output.config(state="normal")
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, text)
        finally:
            if prev_state != str(self.text_output.cget("state")): self.text_output.config(state=prev_state)

    # ==========================================
    # プロンプトエディタ用ロジック (Ver0.19 新機能)
    # ==========================================
    def _reload_editor_list(self):
        self.lb_tasks.delete(0, tk.END)
        path = resolve_path(TASKS_FILENAME)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f: self.raw_tasks_data = json.load(f)
            except: self.raw_tasks_data = {}
        else: self.raw_tasks_data = {}
        # ★ここでもソートして表示
        for key in sorted(self.raw_tasks_data.keys()): self.lb_tasks.insert(tk.END, key)

    def _on_task_select_editor(self, event):
        idx = self.lb_tasks.curselection()
        if not idx: return
        key = self.lb_tasks.get(idx)
        val = self.raw_tasks_data.get(key)
        
        self.current_editing_key = key
        
        self.ent_task_key.delete(0, tk.END)
        self.ent_task_key.insert(0, key)

        self.ent_task_path.delete(0, tk.END)
        self.txt_editor_content.delete("1.0", tk.END)
        
        content = ""
        if isinstance(val, str) and (val.startswith("./") or val.startswith(".\\") or ":/" in val or ":\\" in val):
            self.ent_task_path.insert(0, val)
            real_path = resolve_path(val)
            if real_path and os.path.exists(real_path):
                try:
                    with open(real_path, 'r', encoding='utf-8') as f: content = f.read()
                except Exception as e: content = f"Error reading file: {e}"
            else: content = "(File not found or new file path)"
        else:
            content = val if isinstance(val, str) else str(val)
        
        self.txt_editor_content.insert(tk.END, content)

    def _browse_task_path(self):
        current_val = self.ent_task_path.get().strip()
        initial_dir = get_exe_dir()
        initial_file = ""
        
        if current_val:
            abs_path = resolve_path(current_val)
            if os.path.dirname(abs_path):
                initial_dir = os.path.dirname(abs_path)
            initial_file = os.path.basename(abs_path)
        else:
            # CWDが固定されているので、相対パスの起点(./)は正しい場所になる
            initial_dir = os.path.abspath("./prompts/custom")
            if not os.path.exists(initial_dir):
                try: os.makedirs(initial_dir)
                except: pass

        path = filedialog.asksaveasfilename(
            title="プロンプトファイルの保存先を選択",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            defaultextension=".txt"
        )
        
        if path:
            self.ent_task_path.delete(0, tk.END)
            self.ent_task_path.insert(0, path)

    def _add_new_task(self):
        new_key = simpledialog.askstring("新規", "タスク名(メニュー表示名)を入力:")
        if not new_key: return
        if new_key in self.raw_tasks_data:
            messagebox.showerror("エラー", "その名前は既に存在します")
            return
        
        is_file = messagebox.askyesno("保存形式", "プロンプトを外部テキストファイルとして保存しますか？\n(いいえ=tasks.json内に直接記述)")
        default_path = ""
        default_content = "あなたは優秀なアシスタントです。"
        
        if is_file:
            safe_name = "".join([c for c in new_key if c.isalnum() or c in (' ', '_', '-')]).strip()
            # 相対パスでの保存を推奨（CWD固定により安全）
            rel_path = f"./prompts/custom/{safe_name}.txt"
            default_path = rel_path
            
            # CWD基準でディレクトリ作成
            abs_dir = os.path.abspath("./prompts/custom")
            if not os.path.exists(abs_dir): os.makedirs(abs_dir, exist_ok=True)

        self.raw_tasks_data[new_key] = default_path if is_file else default_content
        
        if is_file:
            abs_path = resolve_path(default_path)
            # ディレクトリがまだなければ作る
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, 'w', encoding='utf-8') as f: f.write(default_content)

        self._save_raw_tasks_json()
        self._reload_editor_list()
        self.lb_tasks.selection_clear(0, tk.END)
        self.lb_tasks.selection_set(tk.END)
        self._on_task_select_editor(None)

    def _save_editor_changes(self):
        new_key = self.ent_task_key.get().strip()
        path_str = self.ent_task_path.get().strip()
        content = self.txt_editor_content.get("1.0", tk.END).strip()
        
        if not new_key:
            messagebox.showwarning("警告", "タスク名を入力してください")
            return

        old_key = self.current_editing_key
        is_rename = (old_key is not None) and (new_key != old_key)

        if is_rename and new_key in self.raw_tasks_data:
            if not messagebox.askyesno("確認", f"タスク名 '{new_key}' は既に存在します。\n上書きしますか？"):
                return

        if path_str:
            real_path = resolve_path(path_str)
            try:
                os.makedirs(os.path.dirname(real_path), exist_ok=True)
                with open(real_path, 'w', encoding='utf-8') as f: f.write(content)
                self.raw_tasks_data[new_key] = path_str
                messagebox.showinfo("保存", f"ファイルを更新しました: {path_str}")
            except Exception as e:
                messagebox.showerror("エラー", f"ファイル保存失敗: {e}")
                return
        else:
            self.raw_tasks_data[new_key] = content
            messagebox.showinfo("保存", "JSON内のプロンプトを更新しました")

        if is_rename and old_key in self.raw_tasks_data:
            del self.raw_tasks_data[old_key]
            self.current_editing_key = new_key

        self._save_raw_tasks_json()
        self._reload_editor_list()
        
        try:
            items = self.lb_tasks.get(0, tk.END)
            if new_key in items:
                idx = items.index(new_key)
                self.lb_tasks.selection_clear(0, tk.END)
                self.lb_tasks.selection_set(idx)
                self.lb_tasks.see(idx)
        except: pass

    def _delete_task(self):
        idx = self.lb_tasks.curselection()
        if not idx: return
        key = self.lb_tasks.get(idx)
        if messagebox.askyesno("削除", f"本当に '{key}' を削除しますか？\n(参照先ファイルは削除されません)"):
            del self.raw_tasks_data[key]
            self._save_raw_tasks_json()
            self._reload_editor_list()
            self.ent_task_key.delete(0, tk.END)
            self.ent_task_path.delete(0, tk.END)
            self.txt_editor_content.delete("1.0", tk.END)

    def _save_raw_tasks_json(self):
        try:
            path = resolve_path(TASKS_FILENAME)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.raw_tasks_data, f, indent=4, ensure_ascii=False)
            # メイン画面用データもリロード
            self.system_prompts, _ = load_system_prompts()
        except Exception as e:
            messagebox.showerror("保存エラー", str(e))

    def _on_tab_changed(self, event):
        # タブ切り替え時にメイン画面のプリセットリストを最新化
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        if tab_text == "テキスト生成":
            current_selection = self.combo_task.get()
            self.combo_task['values'] = sorted(list(self.system_prompts.keys()))
            if current_selection in self.system_prompts:
                self.combo_task.set(current_selection)
            else:
                self.combo_task.current(0) if self.combo_task['values'] else None

    # Appクラス内に追加 (Ver.0.20)
    def _open_folder(self, path=None):
        """指定されたパス（なければCWD）をエクスプローラで開く"""
        if not path:
            # CWDは setup_working_directory で固定されているのでそのまま使う
            path = os.getcwd()
        
        if not os.path.exists(path):
             path = os.getcwd()
            
        try:
            os.startfile(path)
        except Exception as e:
            messagebox.showerror("エラー", f"フォルダを開けませんでした: {e}")

    # Appクラス内に追加 (Ver.0.20)
    def _open_current_task_folder(self):
        path_str = self.ent_task_path.get().strip()
        target_dir = ""
        
        if path_str:
            # 入力がある場合、そのファイルの親ディレクトリを開く
            abs_path = resolve_path(path_str)
            target_dir = os.path.dirname(abs_path)
        else:
            # 入力がない場合、標準のカスタムプロンプトフォルダを開く
            # CWD固定済みなので、abspathで正しい絶対パスになる
            target_dir = os.path.abspath("./prompts/custom")
            if not os.path.exists(target_dir):
                try: os.makedirs(target_dir)
                except: pass
        
        self._open_folder(target_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug: DEBUG_MODE = True
    
    # 【追加】 アプリ起動前に作業ディレクトリを固定
    setup_working_directory()
    
    app = App()
    app.mainloop()
    