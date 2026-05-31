# Kokoro TTS Studio

A local browser-based Text-to-Speech UI powered by **Kokoro TTS** and **Gradio**.

This app lets you paste text, select language/voice/settings, generate natural-sounding audio locally, preview it in the browser, and download the output WAV file.

---

## Features

- Paste plain text and generate speech.
- Select language.
- Select voice.
- Adjust speaking speed.
- Split long text by paragraphs for better narration.
- Preview generated audio in the browser.
- Download generated WAV files.
- Runs locally on your machine.
- No paid API key required.

---

## Recommended Use Cases

- YouTube Shorts voiceovers
- Website/app demo narration
- Product promo videos
- Course narration
- Local TTS testing
- Bulk script-to-audio workflows

---

## Project Structure

```text
kokoro-ui/
  app.py
  README.md
  .gitignore
  outputs/
```

`outputs/` is used for generated audio files and is ignored by Git.

---

## Requirements

Recommended:

- Windows 10/11, macOS, or Linux
- Python 3.10, 3.11, or 3.12
- Python 3.11 is recommended
- `espeak-ng`
- Kokoro TTS
- Gradio

---

## 1. Create Project Folder

```powershell
mkdir kokoro-ui
cd kokoro-ui
```

Place `app.py`, `README.md`, and `.gitignore` in this folder.

---

## 2. Create Python Virtual Environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 3. Install Python Dependencies

```bash
pip install "kokoro>=0.9.4" gradio soundfile numpy
```

Optional, if you later want MP3 export:

```bash
pip install pydub
```

For MP3 export, FFmpeg must also be installed and available in your system PATH.

---

## 4. Install espeak-ng

Kokoro uses `espeak-ng` for phoneme/pronunciation support.

### Windows

1. Download the latest Windows `.msi` installer from the official `espeak-ng` releases page.
2. Run the installer.
3. Restart PowerShell, Command Prompt, or VS Code.
4. Check installation:

```powershell
espeak-ng --version
```

If the command is not recognized, add the `espeak-ng` install folder to your Windows PATH.

### macOS

Using Homebrew:

```bash
brew install espeak-ng
```

### Linux / Ubuntu / Debian

```bash
sudo apt update
sudo apt install espeak-ng
```

---

## 5. Run the App

```bash
python app.py
```

The app should open automatically in your browser.

If it does not open, go to:

```text
http://127.0.0.1:7860
```

---

## 6. Recommended Starting Settings

For app demos and YouTube narration:

```text
Language: English - American
Voice: af_heart or af_bella
Speed: 0.95 to 1.05
Split by paragraphs: enabled
```

Good voices to test first:

```text
af_heart
af_bella
af_sarah
am_adam
am_michael
bf_emma
bm_george
```

---

## 7. Writing Text for Better TTS

Kokoro works best with clean, natural text.

### Good input

```text
Welcome to ReaderNook Lab.

This tool helps you create natural voiceovers from plain text.
Paste your script, choose a voice, and generate clean audio in seconds.
```

### Avoid

```text
Welcome to ReaderNook Lab this tool helps you create natural voiceovers from plain text paste your script choose a voice and generate clean audio in seconds
```

Tips:

- Use short paragraphs.
- Use periods for clear sentence endings.
- Use commas for small pauses.
- Avoid very long paragraphs.
- Avoid unnecessary symbols or formatting.
- For longer videos, split scripts into sections.

---

## 8. Output Files

Generated audio files are saved inside:

```text
outputs/
```

The default format is:

```text
.wav
```

WAV is recommended for editing because it preserves quality.

You can convert WAV to MP3 later using FFmpeg, Audacity, or a Python script with `pydub`.

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'kokoro'`

Make sure your virtual environment is activated, then run:

```bash
pip install "kokoro>=0.9.4"
```

---

### `espeak-ng` not found

Check:

```bash
espeak-ng --version
```

If it fails:

- Install `espeak-ng`.
- Restart your terminal.
- On Windows, add the install location to PATH.

---

### App does not open in browser

Open this manually:

```text
http://127.0.0.1:7860
```

---

### Audio sounds rushed

Try:

```text
Speed: 0.90 to 0.95
Split by paragraphs: enabled
```

Also split long text into shorter paragraphs.

---

### Audio generation is slow

Try:

- Shorter text chunks.
- Fewer paragraphs at a time.
- Close other heavy applications.
- Test one voice at a time.

---

### Some voices fail

Voice availability can vary by Kokoro version and installed dependencies.

Try a known English voice first:

```text
af_heart
af_bella
am_adam
```

Then add/test more voices gradually.

---

## 10. Suggested Future Enhancements

Useful next upgrades:

- MP3 export
- Batch text file input
- Voice preview samples
- Character count and estimated duration
- Output filename field
- Volume normalization
- Local FastAPI endpoint
- Multiple audio export formats
- Presets for YouTube Shorts, course narration, and product demos

---

## 11. Git Notes

Generated files are ignored by `.gitignore`, including:

```text
outputs/
*.wav
*.mp3
*.flac
*.ogg
```

This keeps your Git repository clean and avoids accidentally committing large audio files.

---

## License

This project is a local UI wrapper around Kokoro TTS and Gradio.

Before selling, distributing, or embedding this in a commercial product, review the licenses of:

- Kokoro TTS
- Kokoro model weights
- Gradio
- espeak-ng
- Any additional voices/models you add later
