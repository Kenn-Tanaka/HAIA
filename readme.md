# HAIA (Hybrid AI Assistant)

HAIA is a desktop AI assistant that provides a unified interface
for multiple large language models (LLMs), including:

- Google Gemini
- OpenAI
- Anthropic (Claude)
- OpenRouter
- Local LLMs (Ollama, LM Studio)

In addition to text generation and analysis,
HAIA supports advanced text-to-speech (TTS) with automatic role assignment
("Auto-Casting") using Google Gemini voice models.

---

## Key Features

- **Multi-Model Support**  
  Switch between cloud-based and local LLMs from a single interface.

- **Prompt Management**  
  Create, edit, and manage system prompts (tasks) via GUI.

- **Advanced Text-to-Speech**  
  Automatically assigns voices to characters and generates expressive audio
  (e.g., drama-style playback).

- **Document Processing**  
  Import and export Word, PDF, Markdown, and PowerPoint files via Pandoc.

---

## Intended Use and Limitations

HAIA is designed as a **support and assistance tool**.

- It does not replace human judgment or professional expertise.
- Generated outputs may be inaccurate, incomplete, or speculative.
- All results must be reviewed and validated by users.

HAIA must not be used as a sole authority for:
- Legal or compliance decisions
- Medical or safety-critical judgments
- Security incident or accident investigations
- Official reports or public disclosures

---

## Installation and Usage

HAIA can be used in two ways:

### 1. Python Version (From Source)

- Python 3.10 or later required
- Optional tools:
  - FFmpeg (for MP3 audio output)
  - Pandoc (for document import/export)

See the Japanese README for detailed setup steps.

---

### 2. Standalone Executable (Nuitka Build)

- Built with Nuitka (`--standalone`)
- Python runtime and dependencies are bundled
- No Python installation required

The application automatically detects its data directory
based on the location of configuration files (`*.json`).

---

## Documentation

- **User Manual (Japanese)**  
  See `users_manual.pdf` for detailed usage instructions.

- **Disclaimer (English)**  
  See `DISCLAIMER.md`

- **Corporate / Enterprise Use**  
  See `CORPORATE_USE.md`

---

## License

HAIA is released under the **MIT License**  
and provided on an **"AS IS"** basis, without warranty of any kind.

See `LICENSE.txt` for details.

---

## Third-Party Software

HAIA depends on multiple open-source libraries and external tools.
See `THIRD_PARTY_LICENSES.txt` for a complete list and license information.
