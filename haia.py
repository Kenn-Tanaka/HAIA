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
import platform
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
import xml.etree.ElementTree as ET
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

def resolve_path(filename):
    """設定ファイルのパスを解決する（CWD基準の絶対パスを返す）"""
    return os.path.abspath(filename)

def get_executable_name(base_name):
    """
    OSに応じて実行ファイル名を決定する
    Windows: base_name + .exe
    Linux/Mac: base_name
    """
    name = base_name.lower()
    if platform.system() == "Windows":
        return f"{name}.exe"
    return name

def setup_working_directory():
    """
    tasks.json をアンカーとしてルートディレクトリを特定し、
    カレントディレクトリ(CWD)をそこに固定する。
    """
    base_dir = get_exe_dir()
    candidates = [base_dir, os.path.dirname(base_dir)]
    target_dir = base_dir 

    anchor_file = "tasks.json"

    for d in candidates:
        if os.path.exists(os.path.join(d, anchor_file)):
            target_dir = d
            break
            
    os.chdir(target_dir)
    
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

def find_external_tool(filename):
    """外部ツール(ffmpeg, pandoc)を探索する"""
    cwd_path = os.path.abspath(filename)
    if os.path.exists(cwd_path): return cwd_path
    
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
    target_pandoc = get_executable_name("pandoc")
    pandoc_path = find_external_tool(target_pandoc)
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

# Youtube Transcript
try:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        YOUTUBE_TRANSCRIPT_AVAILABLE = True
    except Exception:
        YOUTUBE_TRANSCRIPT_AVAILABLE = False
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = None

# ==========================================
# 0. 定数・設定・ヘルパー
# ==========================================
APP_VERSION = "0.30"
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
    "temperature": 0.7, "request_timeout": 120, "no_output_timeout": 60, "max_display_lines": 5000,
    "last_model_index": 0, "last_task_index": 0,
    "custom_system_prompt": "あなたは優秀なAIアシスタントです。",
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
        target_ffmpeg = get_executable_name("ffmpeg")
        self.ffmpeg_path = find_external_tool(target_ffmpeg)
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

    def _gen_gemini_dialogue(self, text, voice_map, config_obj, is_script=True, debug_callback=None):
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

        if debug_callback:
            debug_callback(f"--- [Gemini TTS Prompt Request] ---\n{prompt}\n----------------------------------\n")

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

    def save_to_file(self, text, voice_map, is_script, filepath, config_obj, status_callback=None, debug_callback=None):
        if not self.is_ready():
            if status_callback: status_callback("エラー: API Key未設定")
            return
        threading.Thread(target=self._run_save, args=(text, voice_map, is_script, filepath, config_obj, status_callback, False, debug_callback), daemon=True).start()

    def generate_and_play(self, text, voice_map, is_script, config_obj, status_callback=None, debug_callback=None):
        if not self.is_ready():
            if status_callback: status_callback("エラー: API Key未設定")
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_name = tmp.name
        threading.Thread(target=self._run_save, args=(text, voice_map, is_script, tmp_name, config_obj, status_callback, True, debug_callback), daemon=True).start()

    def _run_save(self, text, voice_map, is_script, filepath, config_obj, status_callback, auto_play=False, debug_callback=None):
        try:
            if status_callback: status_callback("音声生成中 (Gemini TTS)...")
            wav_data = self._gen_gemini_dialogue(text, voice_map, config_obj, is_script=is_script, debug_callback=debug_callback)
            
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
# 3. GUIコンポーネント (拡張クラス)
# ==========================================
class ZoomableText(tk.Frame):
    """
    ScrolledTextをFrameベースで再実装し、pack/gridの競合を解消したクラス。
    フォントサイズ変更(Zoom)、右クリックメニュー、
    および「折り返しなし」時の自動水平スクロールバー表示に対応。
    """
    def __init__(self, parent, font_obj, **kwargs):
        super().__init__(parent) # Frameとして初期化
        self.font_obj = font_obj
        
        # wrap引数を抽出（Textウィジェット用）
        wrap_mode = kwargs.pop("wrap", tk.CHAR)
        
        # Gridの設定 (このFrame内部のレイアウト)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # ウィジェットの作成
        self.vbar = ttk.Scrollbar(self, orient="vertical")
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal")
        
        # Textウィジェット
        self.text = tk.Text(self, font=font_obj, wrap=wrap_mode,
                            yscrollcommand=self.vbar.set,
                            xscrollcommand=self.h_scroll.set,
                            **kwargs)
        
        self.vbar.config(command=self.text.yview)
        self.h_scroll.config(command=self.text.xview)
        
        # 初期配置
        self._update_layout(wrap_mode)
        
        # イベントバインド (Textウィジェットに対して行う)
        self.text.bind("<Control-MouseWheel>", self._on_wheel)
        self.text.bind("<Button-3>", self._show_menu)
        
    def _on_wheel(self, event):
        """Ctrl+マウスホイールでフォントサイズを変更"""
        current_size = self.font_obj.cget("size")
        delta = 1 if event.delta > 0 else -1
        new_size = current_size + delta
        if 6 <= new_size <= 40:
            self.font_obj.configure(size=new_size)
        return "break"

    def set_wrap(self, mode):
        """折り返しモードを変更し、レイアウトを更新"""
        self.text.configure(wrap=mode)
        self._update_layout(mode)

    def _update_layout(self, mode):
        """内部ウィジェットのGrid配置を更新"""
        # 一旦配置をクリア
        self.text.grid_forget()
        self.vbar.grid_forget()
        self.h_scroll.grid_forget()
        
        # テキスト: 全面
        self.text.grid(row=0, column=0, sticky="nsew")
        
        # 垂直バー: 右側
        self.vbar.grid(row=0, column=1, sticky="ns")
        
        # 水平バー: 下側（折り返し無しの時のみ）
        if mode == "none":
            self.h_scroll.grid(row=1, column=0, sticky="ew")

    def _show_menu(self, event):
        """右クリックメニューを表示"""
        m = Menu(self, tearoff=0)
        # 修正: event_generate の対象を self.text に変更
        m.add_command(label="Cut", command=lambda: self.text.event_generate("<<Cut>>"))
        m.add_command(label="Copy", command=lambda: self.text.event_generate("<<Copy>>"))
        m.add_command(label="Paste", command=lambda: self.text.event_generate("<<Paste>>"))
        m.add_separator()
        m.add_command(label="Select All", command=self._select_all)
        m.add_separator()
        
        current_wrap = self.text.cget("wrap")
        if current_wrap == "none":
            m.add_command(label="折り返し: ONにする", command=lambda: self.set_wrap("word"))
        else:
            m.add_command(label="折り返し: OFFにする (横スクロール有効)", command=lambda: self.set_wrap("none"))
            
        m.add_command(label="フォントサイズ初期化(10)", command=lambda: self.font_obj.configure(size=10))
        m.add_separator()
        m.add_command(label="内容をクリア", command=self._clear_text)
        
        m.tk_popup(event.x_root, event.y_root)

    def _select_all(self):
        self.text.focus_set()
        self.text.tag_add("sel", "1.0", "end")
        
    def _clear_text(self):
        self.text.delete("1.0", tk.END)
    
    # 修正: config/configure/cget を内部のTextに委譲
    def configure(self, **kwargs):
        try:
            self.text.configure(**kwargs)
        except tk.TclError:
            super().configure(**kwargs)
    
    def config(self, **kwargs):
        self.configure(**kwargs)
        
    def cget(self, key):
        try:
            return self.text.cget(key)
        except tk.TclError:
            return super().cget(key)

    def __getattr__(self, name):
        """Textウィジェットへのメソッド呼び出しを委譲 (insert, get, delete, see等)"""
        if hasattr(self.text, name):
            return getattr(self.text, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# ==========================================
# 4. GUIアプリケーション
# ==========================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"HAIA Ver.{APP_VERSION}" + (" [DEBUG]" if DEBUG_MODE else ""))
        
        self.config_manager = ConfigManager()
        self.settings = self.config_manager.load()
        
        self.cloud_models = {} 
        self.system_prompts = {}
        
        self.casting_config = load_casting_config()
        
        self.geometry(f"{self.settings.get('window_width', 900)}x{self.settings.get('window_height', 950)}")
        
        self.handler = LLMHandler()
        self.audio = AudioHandler()
        self.dynamic_model_map = {} 
        self.msg_queue = queue.Queue()
        self.output_log_chunks = []
        self.max_display_lines = int(self.settings.get("max_display_lines", 5000))
        self.no_output_timeout = float(self.settings.get("no_output_timeout", 60))
        self.is_running = False
        self.base_font_size = self.settings.get("font_size", 10)
        
        self.custom_font = tkfont.Font(family="Meiryo UI", size=self.base_font_size)
        
        self._create_menu()
        self._init_ui()
        
        self.after(500, lambda: self._refresh_model_list(silent=True))
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(1000, self._check_gemini_key)

    def on_close(self):
        self.audio.stop_playback()
        self.settings.update({
            "window_width": self.winfo_width(),
            "window_height": self.winfo_height(),
            "font_size": self.custom_font.cget("size"), 
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

    def _init_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self.tab_main = ttk.Frame(self.notebook)
        self.tab_editor = ttk.Frame(self.notebook)
        self.tab_log = ttk.Frame(self.notebook) 

        self.notebook.add(self.tab_main, text="テキスト生成") 
        self.notebook.add(self.tab_editor, text="プロンプト管理")
        self.notebook.add(self.tab_log, text="ログ") 

        self._init_main_tab()
        self._init_editor_tab()
        self._init_log_tab() 
        
        self.bind('<Control-Return>', lambda e: self.on_run())
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _init_main_tab(self):
        control_frame = ttk.LabelFrame(self.tab_main, text="AI設定", padding=(10, 5, 10, 10))
        control_frame.pack(fill="x", padx=10, pady=5)
        f_upper_control = ttk.Frame(control_frame, padding=(0, 0, 0, 0))
        f_lower_control = ttk.Frame(control_frame, padding=(0, 0, 0, 0))
        f_upper_control.pack(side="top", fill="x", anchor="w")
        f_lower_control.pack(side="top", fill="x", anchor="w")
        f_left_control = ttk.Frame(f_upper_control, padding=(0, 0, 0, 0))
        f_right_control = ttk.Frame(f_upper_control, padding=(0, 0, 0, 0))
        f_left_control.pack(side="left", fill="x")
        f_right_control.pack(side="left", fill="x", padx=25)

        f_model = ttk.Frame(f_left_control, padding=(0, 0, 0, 0))
        f_model.pack(side="top", fill="x", pady=2)
        ttk.Label(f_model, text="モデル:").pack(side="left")
        self.combo_model = ttk.Combobox(f_model, state="readonly", width=60)
        self.combo_model.pack(side="left", padx=5)
        self.btn_model_refresh = ttk.Button(f_model, text="更新", width=4, command=lambda: self._refresh_model_list(False))
        self.btn_model_refresh.pack(side="left")

        f_prompt_ctrl = ttk.Frame(f_left_control, padding=(0, 0, 0, 0))
        f_prompt_ctrl.pack(fill="x", pady=(10, 2))
        ttk.Label(f_prompt_ctrl, text="プリセット:").pack(side="left")
        
        initial_tasks = sorted(list(self.system_prompts.keys())) if self.system_prompts else []
        self.combo_task = ttk.Combobox(f_prompt_ctrl, values=initial_tasks, state="readonly", width=70)
        self.combo_task.pack(side="left", padx=5)
        self.combo_task.bind("<<ComboboxSelected>>", self._on_preset_selected)

        f_temp = ttk.Frame(f_right_control, padding=(0, 0, 0, 0))
        f_temp.pack(side="left", fill="x", pady=2)
        ttk.Label(f_temp, text="創造性 (Temperature):").pack(side="top", anchor="w")
        self.scale_temp = tk.Scale(f_temp, from_=0.0, to=1.0, resolution=0.1, orient="horizontal", length=200)
        self.scale_temp.set(self.settings.get("temperature", 0.7))
        self.scale_temp.pack(side="top", anchor="w", padx=10)
       
        ttk.Label(f_lower_control, text="システムプロンプト (編集可 - 一時的):").pack(anchor="w", pady=(5, 0))
        self.text_system_prompt = ZoomableText(f_lower_control, font_obj=self.custom_font, height=5)
        self.text_system_prompt.pack(fill="both", expand=True, padx=5, pady=5)
        self.text_system_prompt.insert("1.0", self.settings.get("custom_system_prompt", ""))

        input_container = ttk.LabelFrame(self.tab_main, text="入力テキスト", padding=(10, 5, 10, 10))
        input_container.pack(fill="both", expand=True, padx=10, pady=5)
        tb_in = ttk.Frame(input_container)
        tb_in.pack(fill="x")
        ttk.Button(tb_in, text="📂 読込", command=self._load_file_to_input).pack(side="left")
        ttk.Button(tb_in, text="Youtube字幕読み込み", command=self._open_youtube_transcript_dialog).pack(side="left", padx=5)
        ttk.Button(tb_in, text="🗑️ クリア", command=lambda: self.text_input.delete("1.0", tk.END)).pack(side="right")
        
        self.text_input = ZoomableText(input_container, font_obj=self.custom_font, height=6)
        self.text_input.pack(fill="both", expand=True)

        self.btn_run = ttk.Button(self.tab_main, text="▶ テキスト生成 (Ctrl+Enter)", command=self.on_run)
        self.btn_run.pack(pady=5, ipadx=40, ipady=5)

        output_container = ttk.LabelFrame(self.tab_main, text="AIの回答 / 音声操作", padding=(10, 5, 10, 10))
        output_container.pack(fill="both", expand=True, padx=10, pady=5)
        tb_out = ttk.Frame(output_container)
        tb_out.pack(fill="x")
        
        self.lbl_engine_status = ttk.Label(tb_out, text="TTS: 準備中", foreground="gray")
        self.lbl_engine_status.pack(side="left")

        # XML出力時に最終結果のみをコピー/保存する
        self.var_export_final_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(tb_out, text="XML最終のみ", variable=self.var_export_final_only).pack(side="left", padx=6)

        ttk.Button(tb_out, text="💾 保存", command=self._save_output).pack(side="left", padx=5)
        ttk.Button(tb_out, text="📋 コピー", command=self._copy_output).pack(side="left")
        ttk.Separator(tb_out, orient="vertical").pack(side="left", padx=5, fill="y")
        ttk.Button(tb_out, text="🔊 自動配役＆再生", command=self._play_audio_with_casting).pack(side="left")
        ttk.Button(tb_out, text="⏹", width=3, command=self._stop_audio).pack(side="left")
        ttk.Button(tb_out, text="🎵 自動配役＆保存", command=self._save_audio_with_casting).pack(side="left", padx=5)
        
        ttk.Button(tb_out, text="⬇️ 入力→出力", command=self._copy_input_to_output).pack(side="right", padx=2)
        ttk.Button(tb_out, text="🔁 入替", command=self._swap_input_output).pack(side="right", padx=2)
        ttk.Button(tb_out, text="⬆️ 転記", command=self._transfer_output).pack(side="right", padx=2)
        
        self.text_output = ZoomableText(output_container, font_obj=self.custom_font, height=10, state="disabled", bg="#f8f8f8")
        self.text_output.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.tab_main, textvariable=self.status_var, relief="sunken").pack(side="bottom", fill="x")

    def _init_editor_tab(self):
        self.current_editing_key = None  

        paned = ttk.PanedWindow(self.tab_editor, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=5, pady=5)
        
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

        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        
        f_name = ttk.Frame(right_frame)
        f_name.pack(fill="x", pady=5)
        ttk.Label(f_name, text="タスク名(キー):").pack(side="left")
        self.ent_task_key = ttk.Entry(f_name)
        self.ent_task_key.pack(side="left", fill="x", expand=True, padx=5)
        self._bind_entry_context_menu(self.ent_task_key) 
        
        f_path = ttk.Frame(right_frame)
        f_path.pack(fill="x", pady=5)
        ttk.Label(f_path, text="保存先ファイル:").pack(side="left")
        self.ent_task_path = ttk.Entry(f_path)
        self.ent_task_path.pack(side="left", fill="x", expand=True, padx=5)
        self._bind_entry_context_menu(self.ent_task_path) 

        ttk.Button(f_path, text="...", width=3, command=self._browse_task_path).pack(side="left", padx=2)
        ttk.Button(f_path, text="📂", width=3, command=self._open_current_task_folder).pack(side="left", padx=2)
        ttk.Label(f_path, text="(空白=JSON直埋込)").pack(side="left")

        ttk.Label(right_frame, text="プロンプト内容:").pack(anchor="w")
        
        self.txt_editor_content = ZoomableText(right_frame, font_obj=self.custom_font)
        self.txt_editor_content.pack(fill="both", expand=True)
        
        ttk.Button(right_frame, text="💾 変更を保存", command=self._save_editor_changes).pack(anchor="e", pady=5)
        
        self.raw_tasks_data = {} 
        self._reload_editor_list()

    def _init_log_tab(self):
        """ログタブの初期化"""
        ttk.Label(self.tab_log, text="実行ログ / TTSプロンプトデバッグ:").pack(anchor="w", padx=5, pady=5)
        self.text_log = ZoomableText(self.tab_log, font_obj=self.custom_font)
        self.text_log.pack(fill="both", expand=True, padx=5, pady=5)
        
        # ログクリアボタン
        ttk.Button(self.tab_log, text="ログをクリア", command=lambda: self.text_log.delete("1.0", tk.END)).pack(pady=5)

    def _log_to_tab(self, text):
        """別スレッドからログタブへ追記するためのメソッド"""
        def _update():
            self.text_log.insert(tk.END, text + "\n")
            self.text_log.see(tk.END)
        self.after(0, _update)

    # ==========================================
    # 内部ロジック & イベント
    # ==========================================

    def _bind_entry_context_menu(self, w):
        """Entryウィジェット用の簡易右クリックメニュー"""
        m = Menu(w, tearoff=0)
        m.add_command(label="Cut", command=lambda: w.event_generate("<<Cut>>"))
        m.add_command(label="Copy", command=lambda: w.event_generate("<<Copy>>"))
        m.add_command(label="Paste", command=lambda: w.event_generate("<<Paste>>"))
        m.add_separator()
        m.add_command(label="Select All", command=lambda: w.select_range(0, tk.END))
        w.bind("<Button-3>", lambda e: m.tk_popup(e.x_root, e.y_root))

    def _on_preset_selected(self, event):
        task = self.combo_task.get()
        self.text_system_prompt.delete("1.0", tk.END)
        self.text_system_prompt.insert(tk.END, self.system_prompts.get(task, ""))

    def _check_gemini_key(self):
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
        self.cloud_models, _ = load_models_config()
        self.system_prompts, _ = load_system_prompts() 
        
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
        
        self.combo_task['values'] = sorted(list(self.system_prompts.keys()))
        last_task_idx = self.settings.get("last_task_index", -1)
        if 0 <= last_task_idx < len(self.combo_task['values']):
            self.combo_task.current(last_task_idx)

        if hasattr(self, 'btn_model_refresh'): self.btn_model_refresh.config(state="normal")
        if not silent: self.status_var.set("モデルリスト更新完了")


    def _resolve_system_prompt(self, ref: str, strict: bool = True) -> str:
        """
        workflow用の system prompt 解決（三段階）
        1) 完全一致: self.system_prompts[ref]
        2) コード解決: 先頭英数字コードを抽出し、"F01 " / "F01:" / "F01-" に一致するキーを探索
        3) 曖昧なら ValueError / 見つからなければ KeyError
        """
        if ref in self.system_prompts:
            return self.system_prompts[ref]

        code_match = re.match(r"^([A-Za-z0-9]+)", (ref or "").strip())
        if code_match:
            code = code_match.group(1)
            candidates = []
            for k in self.system_prompts.keys():
                if k == code:
                    candidates.append(k)
                    continue
                if k.startswith(code + " ") or k.startswith(code + ":") or k.startswith(code + "-"):
                    candidates.append(k)
            if len(candidates) == 1:
                return self.system_prompts[candidates[0]]
            if len(candidates) > 1:
                raise ValueError(f"ref '{ref}' is ambiguous: {candidates}")

        if strict:
            raise KeyError(ref)
        return ""

    def _get_full_output_text(self) -> str:
        if getattr(self, "output_log_chunks", None):
            return "".join(self.output_log_chunks)
        return self.text_output.get("1.0", tk.END)


    def _is_xml_output_text(self, text: str) -> bool:
        t = (text or "").lstrip()
        if not t.startswith("<"):
            return False
        # workflowは複数トップレベル要素で厳密XMLにならない場合があるため、簡易判定
        return ("</" in t and ">" in t)

    def _extract_final_result_from_xml(self, text: str) -> str:
        if not text:
            return ""
        # <final_output>...</final_output> があればそれを最終結果として採用（最後に出たものを優先）
        matches = re.findall(r"<final_output>\s*(.*?)\s*</final_output>", text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

        # それ以外は「最後のタグの中身」を粗く拾う（互換・保険）
        m = re.search(r"<([A-Za-z0-9_\-]+)>\s*(.*?)\s*</\1>\s*$", text, flags=re.DOTALL)
        if m:
            return (m.group(2) or "").strip()

        return text

    def _get_output_text_for_export(self) -> str:
        full_text = self._get_full_output_text()
        try:
            if getattr(self, "var_export_final_only", None) and self.var_export_final_only.get():
                if self._is_xml_output_text(full_text):
                    return self._extract_final_result_from_xml(full_text)
        except Exception:
            # 失敗時は全量にフォールバック
            pass
        return full_text

    def _append_output_buffer(self, text_chunk: str) -> None:
        if text_chunk:
            self.output_log_chunks.append(text_chunk)

    def _trim_output_widget_lines(self) -> None:
        max_lines = int(getattr(self, "max_display_lines", 5000))
        try:
            end_idx = self.text_output.index("end-1c")
            current_lines = int(end_idx.split(".")[0])
        except Exception:
            return
        if current_lines > max_lines:
            delete_lines = current_lines - max_lines
            try:
                self.text_output.delete("1.0", f"{delete_lines + 1}.0")
            except Exception:
                pass

    def _stream_with_watchdog(self, stream_iter, stop_flag_callable):
        """
        stream_iter を別スレッドで回し、no_output_timeout 秒無音なら TimeoutError を送出。
        """
        q = queue.Queue()
        sentinel = object()

        def _worker():
            try:
                for ch in stream_iter:
                    if stop_flag_callable():
                        break
                    q.put(("data", ch))
                q.put(("done", None))
            except Exception as e:
                q.put(("err", e))
            finally:
                q.put(("end", sentinel))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        last_time = time.time()
        while True:
            if stop_flag_callable():
                return
            try:
                kind, payload = q.get(timeout=0.25)
            except queue.Empty:
                if (time.time() - last_time) > float(self.no_output_timeout):
                    raise TimeoutError(f"No output for {self.no_output_timeout} seconds")
                continue

            if kind == "data":
                last_time = time.time()
                yield payload
            elif kind == "err":
                raise payload
            elif kind == "done":
                return
            elif kind == "end":
                return

    def _expand_placeholders(self, text_in: str, local_map: dict, child_scopes=None) -> str:
        """
        [[DOCUMENT_X]] / [[STEP_ID]] を置換。ネスト対応時は [[A1.S1]] のみ child_scopes から解決。
        """
        if text_in is None:
            return ""
        child_scopes = child_scopes or {}

        def repl(m):
            key = m.group(1).strip()
            if "." in key:
                return str(child_scopes.get(key, m.group(0)))
            return str(local_map.get(key, m.group(0)))

        return re.sub(r"\[\[([^\]]+)\]\]", repl, text_in)

    def _execute_workflow(self, full_id: str, workflow_xml: str, document_text: str, temperature: float, ui_emit: bool = True, depth: int = 0, max_depth: int = 5):
        if depth > max_depth:
            raise ValueError("workflow nesting is too deep")

        try:
            root = ET.fromstring(workflow_xml.strip())
        except Exception as e:
            raise ValueError(f"workflow XML parse error: {e}")

        if root.tag.lower() != "workflow":
            raise ValueError("root tag must be <workflow>")

        local_map = {"DOCUMENT_X": document_text}
        child_scopes = {}

        steps = list(root.findall("./step"))
        for step in steps:
            if not self.is_running:
                return "", root

            step_id = step.attrib.get("id", "").strip()
            step_type = step.attrib.get("type", "llm").strip().lower()
            ref = (step.attrib.get("ref") or "").strip()

            input_node = step.find("input")
            raw_input = input_node.text if input_node is not None and input_node.text is not None else ""
            user_text = self._expand_placeholders(raw_input, local_map, child_scopes)

            if step_type == "workflow":
                # ネスト：子workflowは step 内の <workflow> を対象とする
                child_wf_node = step.find("workflow")
                if child_wf_node is None:
                    raise ValueError(f"workflow step '{step_id}' has no <workflow> child")
                child_xml = ET.tostring(child_wf_node, encoding="unicode")
                child_final, child_root = self._execute_workflow(
                    full_id=full_id,
                    workflow_xml=child_xml,
                    document_text=user_text,
                    temperature=temperature,
                    ui_emit=False,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
                # 親へは child_final を返す
                local_map[step_id] = child_final
                # 明示参照用スコープ（例: [[A1.S1]]）
                child_scopes[f"{step_id}.final"] = child_final
                try:
                    for cs in child_root.findall(".//step"):
                        cid = (cs.attrib.get("id") or "").strip()
                        if not cid:
                            continue
                        rnode = cs.find("result")
                        if rnode is not None and rnode.text is not None:
                            child_scopes[f"{step_id}.{cid}"] = rnode.text
                except Exception:
                    pass

                # XMLに子結果を埋め込む
                result_node = step.find("result")
                if result_node is None:
                    result_node = ET.SubElement(step, "result")
                result_node.clear()
                result_node.text = child_final

                continue

            # llm step
            if not ref:
                raise KeyError(f"step '{step_id}' has no ref")

            system_prompt = self._resolve_system_prompt(ref, strict=True)
            req_timeout = float(self.settings.get("request_timeout", 120))

            if ui_emit:
                self.msg_queue.put(("data", f"<{step_id}>\n"))

            collected = []
            flush_buf = []
            last_flush = time.time()
            last_flush_len = 0

            stream_iter = self.handler.stream_response(
                full_id, system_prompt, user_text, temperature=temperature, timeout=req_timeout
            )
            for chunk in self._stream_with_watchdog(stream_iter, stop_flag_callable=lambda: not self.is_running):
                if not self.is_running:
                    return {step_id: "".join(collected)}, root
                collected.append(chunk)
                flush_buf.append(chunk)

                now = time.time()
                buf_len = sum(len(x) for x in flush_buf)
                if buf_len >= 800 or (now - last_flush) >= 0.15:
                    if ui_emit:
                        self.msg_queue.put(("data", "".join(flush_buf)))
                    flush_buf.clear()
                    last_flush = now
                    last_flush_len = 0

            if flush_buf and ui_emit:
                self.msg_queue.put(("data", "".join(flush_buf)))

            step_text = "".join(collected)
            local_map[step_id] = step_text

            if ui_emit:
                self.msg_queue.put(("data", f"\n</{step_id}>\n"))

            # XMLに結果を埋め込む（<result>）
            result_node = step.find("result")
            if result_node is None:
                result_node = ET.SubElement(step, "result")
            result_node.text = step_text

        final_output = local_map.get("final_output", "")
        return final_output, root

    def _run_workflow_thread(self, full_id: str, workflow_xml: str, document_text: str, temperature: float):
        try:
            final_text, root = self._execute_workflow(full_id, workflow_xml, document_text, temperature, ui_emit=True)
            if self.is_running:
                self.msg_queue.put(("finish", None))
        except Exception as e:
            if self.is_running:
                self.msg_queue.put(("error", str(e)))

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
        if not display_name:
            return
        full_id = self.dynamic_model_map.get(display_name, "")

        sys_prompt = self.text_system_prompt.get("1.0", tk.END).strip()
        temperature = self.scale_temp.get()

        self.is_running = True
        self.btn_run.config(text="■ 中断")

        # 出力初期化（全量バッファも）
        self.output_log_chunks = []
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)

        self.status_var.set(f"生成中... (Temp: {temperature})")
        self.notebook.select(self.tab_main)

        if sys_prompt.lstrip().lower().startswith("<workflow"):
            thread = threading.Thread(
                target=self._run_workflow_thread,
                args=(full_id, sys_prompt, input_text, temperature),
                daemon=True
            )
        else:
            thread = threading.Thread(
                target=self._run_thread,
                args=(full_id, sys_prompt, input_text, temperature),
                daemon=True
            )

        thread.start()
        self.after(50, self._process_queue)

    def _run_thread(self, full_id, sys_prompt, text, temp):
        try:
            req_timeout = float(self.settings.get("request_timeout", 120))
            stream_iter = self.handler.stream_response(full_id, sys_prompt, text, temperature=temp, timeout=req_timeout)
            for chunk in self._stream_with_watchdog(stream_iter, stop_flag_callable=lambda: not self.is_running):
                if not self.is_running:
                    return
                self.msg_queue.put(("data", chunk))
            if self.is_running:
                self.msg_queue.put(("finish", None))
        except Exception as e:
            if self.is_running:
                self.msg_queue.put(("error", str(e)))

    def _process_queue(self):
        """
        UIフリーズ対策:
        - ("data", ...) をまとめ書き
        - see() は最後に1回
        - 1回の処理時間に上限を設ける
        - finish/error を優先
        """
        start_time = time.perf_counter()
        data_buf = []
        finish_msg = None
        error_msg = None

        while (time.perf_counter() - start_time) < 0.03:
            try:
                m_type, content = self.msg_queue.get_nowait()
            except queue.Empty:
                break

            if m_type == "data":
                if content:
                    data_buf.append(content)
            elif m_type == "finish":
                finish_msg = "完了"
                break
            elif m_type == "error":
                error_msg = content or "unknown error"
                break

        if data_buf:
            joined = "".join(data_buf)
            self._append_output_buffer(joined)
            try:
                self.text_output.insert(tk.END, joined)
                self._trim_output_widget_lines()
                self.text_output.see(tk.END)
            except Exception:
                pass

        if error_msg is not None:
            err_text = f"\n[Error] {error_msg}"
            self._append_output_buffer(err_text)
            try:
                self.text_output.insert(tk.END, err_text)
                self.text_output.see(tk.END)
            except Exception:
                pass
            self._end_proc("エラー発生")
            return

        if finish_msg is not None:
            self._end_proc(finish_msg)
            return

        if self.is_running:
            self.after(50, self._process_queue)
        else:
            self._end_proc("中断")

    def _end_proc(self, msg):
        self.is_running = False
        self.btn_run.config(text="▶ テキスト生成 (Ctrl+Enter)")
        self.text_output.config(state="disabled")
        self.status_var.set(msg)

    def _make_auto_filename(self, ext):
        base_name = self.combo_task.get()
        if not base_name: base_name = "Output"
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
        text = self._get_full_output_text().strip()
        if not text: return
        threading.Thread(target=self._run_cast_and_play, args=(text,), daemon=True).start()

    def _run_cast_and_play(self, text):
        voice_map, is_script = self._perform_casting(text)
        if voice_map:
            self.audio.generate_and_play(
                text, voice_map, is_script, self.casting_config, 
                status_callback=lambda m: self.status_var.set(m),
                debug_callback=self._log_to_tab
            )

    def _save_audio_with_casting(self):
        text = self._get_full_output_text().strip()
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
            self.audio.save_to_file(
                text, voice_map, is_script, filename, self.casting_config, 
                status_callback=cb,
                debug_callback=self._log_to_tab
            )

    def _stop_audio(self):
        self.audio.stop_playback()
        self.status_var.set("再生停止")
    
    # --- 設定管理 ---
    def _reload_configs(self):
        self._refresh_model_list()
        self.system_prompts, _ = load_system_prompts()
        self.casting_config = load_casting_config()
        self.combo_task['values'] = sorted(list(self.system_prompts.keys()))
        self._reload_editor_list()
        messagebox.showinfo("更新", "設定を再読み込みしました")

    def _change_font_size(self, d):
        """全体のフォントサイズを変更（ZoomableTextで共有されているため全てに反映される）"""
        new_s = self.custom_font.cget("size") + d
        if 6 <= new_s <= 40:
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

    # ------------------------------
    # YouTubeタイトル・字幕読み込み（Ver.0.30+）
    # ------------------------------
    def _yt_get_video_info(self, url):
        """oEmbed APIを使用して動画のタイトルを取得する"""
        try:
            # YouTubeのoEmbedエンドポイントを利用
            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
            res = requests.get(oembed_url, timeout=5)
            if res.status_code == 200:
                return res.json().get("title", "Unknown Title")
        except:
            pass
        return "Unknown Title"

    def _open_youtube_transcript_dialog(self):
        if not globals().get("YOUTUBE_TRANSCRIPT_AVAILABLE", False):
            messagebox.showerror("YouTube字幕", "youtube-transcript-api が見つかりません。")
            return

        dlg = tk.Toplevel(self)
        dlg.title("YouTube字幕読み込み")
        dlg.transient(self)
        dlg.grab_set()

        # URL入力
        frm_url = ttk.Frame(dlg)
        frm_url.pack(fill="x", padx=10, pady=10)
        ttk.Label(frm_url, text="YouTube URL:").pack(side="left")
        url_var = tk.StringVar(value="")
        ent = ttk.Entry(frm_url, textvariable=url_var, width=70)
        ent.pack(side="left", padx=6, fill="x", expand=True)
        self._bind_entry_context_menu(ent) # 右クリックペーストを有効化

        # 操作ボタン
        frm_btn = ttk.Frame(dlg)
        frm_btn.pack(fill="x", padx=10, pady=(0, 8))
        btn_fetch = ttk.Button(frm_btn, text="字幕一覧取得")
        btn_fetch.pack(side="left")
        status_var = tk.StringVar(value="")
        ttk.Label(frm_btn, textvariable=status_var).pack(side="left", padx=10)

        # 一覧（自由選択）
        columns = ("kind", "code", "name", "translatable")
        tree = ttk.Treeview(dlg, columns=columns, show="headings", height=10)
        # ... (Treeviewの見出し設定は以前と同じ) ...
        tree.heading("kind", text="種別")
        tree.heading("code", text="言語コード")
        tree.heading("name", text="言語名")
        tree.heading("translatable", text="翻訳可")
        tree.column("kind", width=90, anchor="w")
        tree.column("code", width=90, anchor="w")
        tree.column("name", width=260, anchor="w")
        tree.column("translatable", width=70, anchor="center")
        tree.pack(fill="both", expand=True, padx=10)

        # スクロール
        sc = ttk.Scrollbar(dlg, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sc.set)
        sc.place(in_=tree, relx=1.0, rely=0, relheight=1.0, anchor="ne")

        # 下部ボタン
        frm_bottom = ttk.Frame(dlg)
        frm_bottom.pack(fill="x", padx=10, pady=(0, 10))
        btn_load = ttk.Button(frm_bottom, text="読み込み", state="disabled")
        btn_load.pack(side="right", padx=(6, 0))
        btn_cancel = ttk.Button(frm_bottom, text="キャンセル", command=dlg.destroy)
        btn_cancel.pack(side="right")

        api = YouTubeTranscriptApi()
        transcript_map = {}

        def set_busy(is_busy: bool, msg: str = ""):
            status_var.set(msg)
            try:
                btn_fetch.config(state=("disabled" if is_busy else "normal"))
                btn_load.config(state=("disabled" if is_busy else ("normal" if tree.selection() else "disabled")))
                btn_cancel.config(state=("disabled" if is_busy else "normal"))
                ent.config(state=("disabled" if is_busy else "normal"))
                tree.config(selectmode=("none" if is_busy else "browse"))
            except Exception: pass

        def on_select(_evt=None):
            sel = tree.selection()
            btn_load.config(state=("normal" if sel else "disabled"))

        tree.bind("<<TreeviewSelect>>", on_select)

        def fetch_list():
            url = url_var.get().strip()
            if not url:
                messagebox.showerror("YouTube字幕", "URLを入力してください。")
                return
            try:
                vid = self._yt_parse_video_id(url)
            except Exception as e:
                messagebox.showerror("YouTube字幕", str(e))
                return

            def worker():
                set_busy(True, "字幕一覧を取得中…")
                try:
                    tlist = api.list(vid)
                    items = []
                    for t in tlist:
                        kind = "自動" if getattr(t, "is_generated", False) else "手動"
                        code = getattr(t, "language_code", "")
                        name = getattr(t, "language", "")
                        trans = "Yes" if getattr(t, "is_translatable", False) else ""
                        items.append((t, kind, code, name, trans))
                    def ui():
                        tree.delete(*tree.get_children())
                        transcript_map.clear()
                        for t, kind, code, name, trans in items:
                            iid = tree.insert("", "end", values=(kind, code, name, trans))
                            transcript_map[iid] = t
                        status_var.set(f"{len(items)} 件")
                        btn_load.config(state="disabled")
                    self.after(0, ui)
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("YouTube字幕", f"取得失敗: {e}"))
                finally:
                    self.after(0, lambda: set_busy(False, status_var.get()))

            threading.Thread(target=worker, daemon=True).start()

        def load_selected():
            sel = tree.selection()
            if not sel: return
            t = transcript_map.get(sel[0])
            url = url_var.get().strip()

            def worker():
                set_busy(True, "データ収集中…")
                try:
                    # タイトルと字幕データを並行して取得
                    title = self._yt_get_video_info(url)
                    rows = t.fetch().to_raw_data() # 最新APIのオブジェクトをリストに変換
                    
                    lines = [f"Title: {title}", f"URL: {url}", "---", "Transcript:"]
                    for r in rows:
                        s = str(r.get("text", "")).strip()
                        if s: lines.append(s)
                    
                    full_text = "\n".join(lines)

                    def ui():
                        self.text_input.delete("1.0", tk.END)
                        self.text_input.insert("1.0", full_text)
                        dlg.destroy()
                    self.after(0, ui)
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Error", str(e)))
                finally:
                    self.after(0, lambda: set_busy(False, status_var.get()))

            threading.Thread(target=worker, daemon=True).start()

        btn_fetch.config(command=fetch_list)
        btn_load.config(command=load_selected)

        # 初期フォーカス
        try:
            ent.focus_set()
        except Exception:
            pass

    def _yt_parse_video_id(self, url: str) -> str:
        # 代表的な形式に対応（watch, youtu.be, shorts, embed）
        u = (url or "").strip()
        if not u:
            raise ValueError("URLが空です。")

        # 文字列中に video id っぽい v= がある場合
        m = re.search(r"[?&]v=([0-9A-Za-z_-]{6,})", u)
        if m:
            return m.group(1)

        # youtu.be/<id>
        m = re.search(r"youtu\.be/([0-9A-Za-z_-]{6,})", u)
        if m:
            return m.group(1)

        # /shorts/<id>
        m = re.search(r"/shorts/([0-9A-Za-z_-]{6,})", u)
        if m:
            return m.group(1)

        # /embed/<id>
        m = re.search(r"/embed/([0-9A-Za-z_-]{6,})", u)
        if m:
            return m.group(1)

        # 最後の手段：ID単体入力
        m = re.fullmatch(r"[0-9A-Za-z_-]{6,}", u)
        if m:
            return u

        raise ValueError("対応していないURL形式です。")

    def _save_output(self):
        if PANDOC_AVAILABLE:
            ft = [("Text File", "*.txt"), ("Markdown", "*.md"), ("Word Document", "*.docx"),
                  ("PowerPoint", "*.pptx"), ("OpenDocument Text", "*.odt"),
                  ("HTML File", "*.html"), ("EPUB Book", "*.epub")]
            d_ext = ".txt"
        else:
            ft = [("Text File", "*.txt")]
            d_ext = ".txt"

        default_filename = self._make_auto_filename(d_ext) 

        fp = filedialog.asksaveasfilename(
            defaultextension=d_ext, filetypes=ft,
            initialfile=default_filename, 
            title="保存先を選択"
        )
        if not fp: return

        try:
            content = self._get_output_text_for_export()
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
        self.clipboard_append(self._get_output_text_for_export())
        self.status_var.set("コピーしました")

    def _transfer_output(self):
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert(tk.END, self._get_full_output_text())

    def _copy_input_to_output(self):
        src = self.text_input.get("1.0", tk.END)
        self._set_output_text(src)
        self.status_var.set("入力テキストをAI出力欄へコピーしました")

    def _swap_input_output(self):
        in_text = self.text_input.get("1.0", tk.END)
        out_text = self._get_full_output_text()
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert(tk.END, out_text)
        self._set_output_text(in_text)
        self.status_var.set("入力/出力テキストを入れ替えました")

    def _set_output_text(self, text: str):
        prev_state = str(self.text_output.cget("state"))
        try:
            if prev_state != "normal": self.text_output.config(state="normal")
            self.output_log_chunks = [text]
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, text)
            self._trim_output_widget_lines()
        finally:
            if prev_state != str(self.text_output.cget("state")): self.text_output.config(state=prev_state)

    # ==========================================
    # プロンプトエディタ用ロジック
    # ==========================================
    def _reload_editor_list(self):
        self.lb_tasks.delete(0, tk.END)
        path = resolve_path(TASKS_FILENAME)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f: self.raw_tasks_data = json.load(f)
            except: self.raw_tasks_data = {}
        else: self.raw_tasks_data = {}
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
            initial_dir = os.path.abspath("./prompts/custom")
            if not os.path.exists(initial_dir):
                try: os.makedirs(initial_dir)
                except: pass

        path = filedialog.asksaveasfilename(
            title="プロンプト保存先パスの選択",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            defaultextension=".txt",
            confirmoverwrite=False 
        )
        
        if path:
            self.ent_task_path.delete(0, tk.END)
            self.ent_task_path.insert(0, path)
            # ここではファイル書き込みを行わず、Entryへの反映のみとする

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
            rel_path = f"./prompts/custom/{safe_name}.txt"
            default_path = rel_path
            
            abs_dir = os.path.abspath("./prompts/custom")
            if not os.path.exists(abs_dir): os.makedirs(abs_dir, exist_ok=True)

        self.raw_tasks_data[new_key] = default_path if is_file else default_content
        
        if is_file:
            abs_path = resolve_path(default_path)
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

        # 既存ファイルへの上書き確認ロジックを追加
        if path_str:
            real_path = resolve_path(path_str)
            if os.path.exists(real_path):
                if not messagebox.askyesno("上書き確認", f"ファイルは既に存在します：\n{path_str}\n\n上書きしてよろしいですか？"):
                    return

            try:
                os.makedirs(os.path.dirname(real_path), exist_ok=True)
                with open(real_path, 'w', encoding='utf-8') as f: f.write(content)
                self.raw_tasks_data[new_key] = path_str
                messagebox.showinfo("保存", f"ファイルを更新しました: {path_str}")
            except Exception as e:
                messagebox.showerror("エラー", f"ファイル保存失敗: {e}")
                return
        else:
            # JSON直接埋め込みの場合
            self.raw_tasks_data[new_key] = content
            messagebox.showinfo("保存", "JSON内のプロンプトを更新しました")

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
            self.system_prompts, _ = load_system_prompts()
        except Exception as e:
            messagebox.showerror("保存エラー", str(e))

    def _on_tab_changed(self, event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        if tab_text == "テキスト生成":
            current_selection = self.combo_task.get()
            self.combo_task['values'] = sorted(list(self.system_prompts.keys()))
            if current_selection in self.system_prompts:
                self.combo_task.set(current_selection)
            else:
                self.combo_task.current(0) if self.combo_task['values'] else None

    def _open_folder(self, path=None):
        """指定されたパス（なければCWD）をエクスプローラで開く"""
        if not path:
            path = os.getcwd()
        
        if not os.path.exists(path):
             path = os.getcwd()
            
        try:
            os.startfile(path)
        except Exception as e:
            messagebox.showerror("エラー", f"フォルダを開けませんでした: {e}")

    def _open_current_task_folder(self):
        path_str = self.ent_task_path.get().strip()
        target_dir = ""
        
        if path_str:
            abs_path = resolve_path(path_str)
            target_dir = os.path.dirname(abs_path)
        else:
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
    
    setup_working_directory()
    
    app = App()
    app.mainloop()
    