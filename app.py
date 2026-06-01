from pathlib import Path
from datetime import datetime
import csv
import io
import re

import gradio as gr
import numpy as np
import soundfile as sf
from kokoro import KPipeline
import torch


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 24000
SETTINGS_STORAGE_KEY = "kokoro-tts-studio-settings-v1"
SETTINGS_SECRET = "kokoro-tts-studio-local-settings"
GENDER_OPTIONS = ["All", "Female", "Male"]
DEFAULT_SETTINGS = {
    "language": "English - American",
    "gender": "All",
    "voice": "af_heart",
    "speed": 1.0,
    "split_by_paragraphs": True,
    "add_sentence_pauses": True,
    "sentence_pause_ms": 250,
    "paragraph_pause_ms": 550,
}
APP_CSS = """
#excel-paste textarea {
    font-family: Consolas, "Courier New", monospace;
    line-height: 1.65;
    tab-size: 18;
    white-space: pre;
    overflow: auto;
    background:
        repeating-linear-gradient(
            to bottom,
            #ffffff 0,
            #ffffff 27px,
            #f7fafc 27px,
            #f7fafc 54px
        );
}

#batch-table .table-wrap,
#results-table .table-wrap {
    border-radius: 6px;
    border: 1px solid #d9dee8;
}

#batch-table table,
#results-table table {
    font-size: 13px;
}

#batch-table thead th,
#results-table thead th {
    background: #f3f6fb;
    color: #1f2937;
    font-weight: 700;
}

#batch-table tbody tr:nth-child(even),
#results-table tbody tr:nth-child(even) {
    background: #fafbfd;
}

#batch-table td,
#results-table td {
    vertical-align: top;
}

.danger-button button {
    border-color: #dc2626 !important;
    color: #dc2626 !important;
}
"""

LANGUAGES = {
    "English - American": "a",
    "English - British": "b",
    "Spanish": "e",
    "French": "f",
    "Hindi": "h",
    "Italian": "i",
    "Portuguese - Brazilian": "p",
    "Mandarin Chinese": "z",
}

# Start with stable/common Kokoro voices.
# Add/remove voices after testing on your machine.
VOICES_BY_LANGUAGE = {
    "English - American": [
        "af_heart",
        "af_bella",
        "af_sarah",
        "af_nicole",
        "af_sky",
        "am_adam",
        "am_michael",
    ],
    "English - British": [
        "bf_emma",
        "bf_isabella",
        "bm_george",
        "bm_lewis",
    ],
    "Spanish": [
        "ef_dora",
        "em_alex",
        "em_santa",
    ],
    "French": [
        "ff_siwis",
    ],
    "Hindi": [
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
    ],
    "Italian": [
        "if_sara",
        "im_nicola",
    ],
    "Portuguese - Brazilian": [
        "pf_dora",
        "pm_alex",
        "pm_santa",
    ],
    "Mandarin Chinese": [
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ],
}

PIPELINES = {}


def get_voice_gender(voice: str) -> str:
    if len(voice) >= 2 and voice[1] == "f":
        return "Female"
    if len(voice) >= 2 and voice[1] == "m":
        return "Male"
    return "All"


def get_voice_choices(language_name: str, gender: str) -> list[str]:
    voices = VOICES_BY_LANGUAGE.get(language_name, [])

    if gender in ("Female", "Male"):
        return [voice for voice in voices if get_voice_gender(voice) == gender]

    return voices


def select_voice(language_name: str, gender: str, current_voice: str | None = None):
    voices = get_voice_choices(language_name, gender)

    if current_voice in voices:
        return current_voice

    return voices[0] if voices else None


def get_pipeline(language_name: str):
    lang_code = LANGUAGES[language_name]

    if lang_code not in PIPELINES:
        # PIPELINES[lang_code] = KPipeline(lang_code=lang_code)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        PIPELINES[lang_code] = KPipeline(
            lang_code=lang_code,
            repo_id="hexgrad/Kokoro-82M",
            device=device,
        )

    return PIPELINES[lang_code]


def normalize_settings(settings):
    merged = DEFAULT_SETTINGS.copy()

    if isinstance(settings, dict):
        merged.update(settings)

    language_name = merged.get("language")
    if language_name not in LANGUAGES:
        language_name = DEFAULT_SETTINGS["language"]

    gender = merged.get("gender")
    if gender not in GENDER_OPTIONS:
        gender = DEFAULT_SETTINGS["gender"]

    speed = merged.get("speed", DEFAULT_SETTINGS["speed"])
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = DEFAULT_SETTINGS["speed"]
    speed = min(1.35, max(0.75, speed))

    split_by_paragraphs = bool(
        merged.get("split_by_paragraphs", DEFAULT_SETTINGS["split_by_paragraphs"])
    )
    add_sentence_pauses = bool(
        merged.get("add_sentence_pauses", DEFAULT_SETTINGS["add_sentence_pauses"])
    )

    sentence_pause_ms = merged.get(
        "sentence_pause_ms",
        DEFAULT_SETTINGS["sentence_pause_ms"],
    )
    paragraph_pause_ms = merged.get(
        "paragraph_pause_ms",
        DEFAULT_SETTINGS["paragraph_pause_ms"],
    )

    try:
        sentence_pause_ms = int(sentence_pause_ms)
    except (TypeError, ValueError):
        sentence_pause_ms = DEFAULT_SETTINGS["sentence_pause_ms"]

    try:
        paragraph_pause_ms = int(paragraph_pause_ms)
    except (TypeError, ValueError):
        paragraph_pause_ms = DEFAULT_SETTINGS["paragraph_pause_ms"]

    sentence_pause_ms = min(1200, max(0, sentence_pause_ms))
    paragraph_pause_ms = min(2000, max(0, paragraph_pause_ms))

    voice = select_voice(language_name, gender, merged.get("voice"))

    return {
        "language": language_name,
        "gender": gender,
        "voice": voice,
        "speed": speed,
        "split_by_paragraphs": split_by_paragraphs,
        "add_sentence_pauses": add_sentence_pauses,
        "sentence_pause_ms": sentence_pause_ms,
        "paragraph_pause_ms": paragraph_pause_ms,
    }


def load_settings(settings):
    settings = normalize_settings(settings)
    voices = get_voice_choices(settings["language"], settings["gender"])

    return (
        settings["language"],
        settings["gender"],
        gr.update(choices=voices, value=settings["voice"]),
        settings["speed"],
        settings["split_by_paragraphs"],
        settings["add_sentence_pauses"],
        settings["sentence_pause_ms"],
        settings["paragraph_pause_ms"],
    )


def save_settings(
    language_name,
    gender,
    voice,
    speed,
    split_by_paragraphs,
    add_sentence_pauses,
    sentence_pause_ms,
    paragraph_pause_ms,
):
    return normalize_settings(
        {
            "language": language_name,
            "gender": gender,
            "voice": voice,
            "speed": speed,
            "split_by_paragraphs": split_by_paragraphs,
            "add_sentence_pauses": add_sentence_pauses,
            "sentence_pause_ms": sentence_pause_ms,
            "paragraph_pause_ms": paragraph_pause_ms,
        }
    )


def update_voice_choices(
    language_name,
    gender,
    current_voice,
    speed,
    split_by_paragraphs,
    add_sentence_pauses,
    sentence_pause_ms,
    paragraph_pause_ms,
):
    selected_voice = select_voice(language_name, gender, current_voice)
    voices = get_voice_choices(language_name, gender)
    settings = save_settings(
        language_name,
        gender,
        selected_voice,
        speed,
        split_by_paragraphs,
        add_sentence_pauses,
        sentence_pause_ms,
        paragraph_pause_ms,
    )

    return gr.update(choices=voices, value=selected_voice), settings


def make_output_path(language_name: str, voice: str, index: int, requested_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if requested_name and str(requested_name).strip():
        base_name = Path(str(requested_name).strip()).stem
    else:
        base_name = f"kokoro_{language_name}_{voice}_{timestamp}_{index:03d}"

    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", base_name)
    safe_name = re.sub(r"\s+", "_", safe_name).strip("._ ")

    if not safe_name:
        safe_name = f"kokoro_{timestamp}_{index:03d}"

    output_path = OUTPUT_DIR / f"{safe_name[:120]}.wav"
    suffix = 2

    while output_path.exists():
        output_path = OUTPUT_DIR / f"{safe_name[:110]}_{suffix}.wav"
        suffix += 1

    return output_path


def sentence_segments(text: str) -> list[str]:
    pieces = re.split(r'([.!?;:।॥]+["\'”’)]*)', text)
    segments = []
    current = ""

    for piece in pieces:
        if not piece:
            continue

        current += piece

        if re.fullmatch(r'[.!?;:।॥]+["\'”’)]*', piece):
            segment = current.strip()
            if segment:
                segments.append(segment)
            current = ""

    remainder = current.strip()
    if remainder:
        segments.append(remainder)

    return segments


def text_segments_with_pauses(
    text: str,
    split_by_paragraphs: bool,
    add_sentence_pauses: bool,
    sentence_pause_ms: int,
    paragraph_pause_ms: int,
):
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    pieces = re.split(r"(\n+)", normalized_text)
    segments = []

    for piece in pieces:
        if not piece:
            continue

        if piece.startswith("\n"):
            if split_by_paragraphs and segments:
                last_text, last_pause_ms = segments[-1]
                segments[-1] = (last_text, max(last_pause_ms, paragraph_pause_ms))
            continue

        text_pieces = sentence_segments(piece) if add_sentence_pauses else [piece.strip()]

        for index, text_piece in enumerate(text_pieces):
            if not text_piece:
                continue

            pause_ms = (
                sentence_pause_ms
                if add_sentence_pauses and index < len(text_pieces) - 1
                else 0
            )
            segments.append((text_piece, pause_ms))

    if segments:
        last_text, _ = segments[-1]
        segments[-1] = (last_text, 0)

    return segments


def synthesize_to_file(
    text: str,
    language_name: str,
    voice: str,
    speed: float,
    split_by_paragraphs: bool,
    add_sentence_pauses: bool,
    sentence_pause_ms: int,
    paragraph_pause_ms: int,
    output_path: Path,
):
    pipeline = get_pipeline(language_name)
    segments = text_segments_with_pauses(
        text,
        split_by_paragraphs,
        add_sentence_pauses,
        sentence_pause_ms,
        paragraph_pause_ms,
    )

    audio_chunks = []

    for segment_text, pause_ms in segments:
        generator = pipeline(
            segment_text,
            voice=voice,
            speed=speed,
            split_pattern=None,
        )

        for _, _, audio in generator:
            audio_chunks.append(audio)

        if pause_ms > 0:
            silence_samples = int(SAMPLE_RATE * pause_ms / 1000)
            audio_chunks.append(np.zeros(silence_samples, dtype=np.float32))

    if not audio_chunks:
        raise gr.Error("No audio was generated. Try shorter text or a different voice.")

    final_audio = np.concatenate(audio_chunks)
    sf.write(output_path, final_audio, SAMPLE_RATE)

    return output_path


def parse_batch_rows(batch_rows):
    rows = []

    if batch_rows is None:
        return rows

    if hasattr(batch_rows, "values"):
        batch_rows = batch_rows.values.tolist()

    if len(batch_rows) == 0:
        return rows

    for row in batch_rows:
        if isinstance(row, dict):
            text = row.get("Text", "")
            requested_name = row.get("File name (optional)", "")
        else:
            text = row[0] if len(row) > 0 else ""
            requested_name = row[1] if len(row) > 1 else ""

        if text is None or not str(text).strip():
            continue

        rows.append((str(text), requested_name))

    return rows


def parse_excel_paste(pasted_rows: str):
    if not pasted_rows or not pasted_rows.strip():
        raise gr.Error("Paste at least one Excel row first.")

    reader = csv.reader(io.StringIO(pasted_rows.strip()), delimiter="\t")
    rows = []

    for row in reader:
        cells = [cell.strip() for cell in row]

        if not cells or not any(cells):
            continue

        if len(rows) == 0:
            normalized = [cell.lower().lstrip("\ufeff") for cell in cells[:2]]
            has_text_header = normalized[0] in {"text", "script", "content"}
            has_file_header = len(normalized) > 1 and normalized[1] in {
                "file",
                "filename",
                "file name",
                "file name (optional)",
            }

            if has_text_header and (len(normalized) == 1 or has_file_header):
                continue

        text = cells[0] if len(cells) > 0 else ""
        requested_name = cells[1] if len(cells) > 1 else ""

        if text:
            rows.append([text, requested_name])

    if not rows:
        raise gr.Error("No text rows found in the pasted data.")

    return rows


def make_preview_choices(output_paths):
    return [(Path(path).name, path) for path in output_paths]


def choose_preview_audio(selected_path):
    if selected_path and Path(selected_path).exists():
        return selected_path

    return None


def delete_output_files():
    deleted_count = 0

    for output_path in OUTPUT_DIR.glob("*.wav"):
        if output_path.is_file():
            output_path.unlink()
            deleted_count += 1

    message = f"Deleted {deleted_count} WAV file{'s' if deleted_count != 1 else ''}."

    return (
        None,
        [],
        "",
        [],
        gr.update(choices=[], value=None),
        message,
    )


def generate_audio(
    text: str,
    language_name: str,
    gender: str,
    voice: str,
    speed: float,
    split_by_paragraphs: bool,
    add_sentence_pauses: bool,
    sentence_pause_ms: int,
    paragraph_pause_ms: int,
):
    if not text or not text.strip():
        raise gr.Error("Please paste some text first.")

    voice = select_voice(language_name, gender, voice)

    if not voice:
        raise gr.Error("Please select a voice.")

    output_path = make_output_path(language_name, voice, 1)
    synthesize_to_file(
        text,
        language_name,
        voice,
        speed,
        split_by_paragraphs,
        add_sentence_pauses,
        sentence_pause_ms,
        paragraph_pause_ms,
        output_path,
    )

    file_paths = [str(output_path)]
    results = [[output_path.name, str(output_path)]]

    return (
        str(output_path),
        file_paths,
        output_path.name,
        results,
        gr.update(choices=make_preview_choices(file_paths), value=str(output_path)),
        "",
    )


def generate_batch_audio(
    batch_rows,
    language_name: str,
    gender: str,
    voice: str,
    speed: float,
    split_by_paragraphs: bool,
    add_sentence_pauses: bool,
    sentence_pause_ms: int,
    paragraph_pause_ms: int,
    progress=gr.Progress(),
):
    rows = parse_batch_rows(batch_rows)

    if not rows:
        raise gr.Error("Please add at least one text row first.")

    voice = select_voice(language_name, gender, voice)

    if not voice:
        raise gr.Error("Please select a voice.")

    output_paths = []
    results = []

    for index, (text, requested_name) in enumerate(
        progress.tqdm(rows, desc="Generating audio files"),
        start=1,
    ):
        output_path = make_output_path(language_name, voice, index, requested_name)
        synthesize_to_file(
            text,
            language_name,
            voice,
            speed,
            split_by_paragraphs,
            add_sentence_pauses,
            sentence_pause_ms,
            paragraph_pause_ms,
            output_path,
        )
        output_paths.append(str(output_path))
        results.append([output_path.name, str(output_path)])

    filename_list = "\n".join(Path(path).name for path in output_paths)

    return (
        output_paths[0],
        output_paths,
        filename_list,
        results,
        gr.update(choices=make_preview_choices(output_paths), value=output_paths[0]),
        "",
    )


with gr.Blocks(title="Kokoro TTS Studio") as demo:
    settings_state = gr.BrowserState(
        DEFAULT_SETTINGS,
        storage_key=SETTINGS_STORAGE_KEY,
        secret=SETTINGS_SECRET,
    )

    gr.Markdown("# Kokoro TTS Studio")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Single Text"):
                    text_input = gr.Textbox(
                        label="Text",
                        placeholder="Paste your narration script here...",
                        lines=12,
                    )

                    generate_btn = gr.Button("Generate Audio", variant="primary")

                with gr.Tab("Batch Rows"):
                    excel_paste = gr.Textbox(
                        label="Excel rows",
                        placeholder="Text\tfile_name",
                        lines=7,
                        elem_id="excel-paste",
                    )

                    load_excel_paste_btn = gr.Button("Load to Table")

                    batch_input = gr.Dataframe(
                        headers=["Text", "File name (optional)"],
                        datatype=["str", "str"],
                        type="array",
                        row_count=25,
                        column_count=2,
                        label="Batch texts",
                        interactive=True,
                        wrap=False,
                        line_breaks=True,
                        max_height=320,
                        column_widths=["72%", "28%"],
                        buttons=["copy"],
                        elem_id="batch-table",
                    )

                    generate_batch_btn = gr.Button("Generate Files", variant="primary")

            with gr.Row(equal_height=True):
                with gr.Column():
                    language = gr.Dropdown(
                        label="Language",
                        choices=list(LANGUAGES.keys()),
                        value=DEFAULT_SETTINGS["language"],
                    )

                    gender = gr.Radio(
                        label="Voice gender",
                        choices=GENDER_OPTIONS,
                        value=DEFAULT_SETTINGS["gender"],
                    )

                with gr.Column():
                    voice = gr.Dropdown(
                        label="Voice",
                        choices=VOICES_BY_LANGUAGE[DEFAULT_SETTINGS["language"]],
                        value=DEFAULT_SETTINGS["voice"],
                        allow_custom_value=True,
                    )

                    speed = gr.Slider(
                        label="Speed",
                        minimum=0.75,
                        maximum=1.35,
                        value=DEFAULT_SETTINGS["speed"],
                        step=0.05,
                    )

                    split_by_paragraphs = gr.Checkbox(
                        label="Pause on line breaks",
                        value=DEFAULT_SETTINGS["split_by_paragraphs"],
                    )

                    add_sentence_pauses = gr.Checkbox(
                        label="Pause after sentences",
                        value=DEFAULT_SETTINGS["add_sentence_pauses"],
                    )

                    with gr.Row():
                        sentence_pause_ms = gr.Slider(
                            label="Sentence pause (ms)",
                            minimum=0,
                            maximum=1200,
                            value=DEFAULT_SETTINGS["sentence_pause_ms"],
                            step=25,
                        )

                        paragraph_pause_ms = gr.Slider(
                            label="Line break pause (ms)",
                            minimum=0,
                            maximum=2000,
                            value=DEFAULT_SETTINGS["paragraph_pause_ms"],
                            step=50,
                        )

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Preview",
                type="filepath",
            )

            preview_selector = gr.Dropdown(
                label="Preview generated file",
                choices=[],
                value=None,
                interactive=True,
            )

            file_output = gr.File(
                label="Download WAV files",
                file_count="multiple",
            )

            generated_names = gr.Textbox(
                label="Generated filenames",
                lines=8,
                interactive=False,
                buttons=["copy"],
            )

            generation_results = gr.Dataframe(
                headers=["File name", "Path"],
                datatype=["str", "str"],
                type="array",
                label="Generated files",
                interactive=False,
                wrap=False,
                max_height=260,
                column_widths=["66%", "34%"],
                buttons=["copy"],
                elem_id="results-table",
            )

            delete_files_btn = gr.Button(
                "Delete Previous WAV Files",
                variant="secondary",
                elem_classes=["danger-button"],
            )

            cleanup_status = gr.Markdown()

    demo.load(
        fn=load_settings,
        inputs=settings_state,
        outputs=[
            language,
            gender,
            voice,
            speed,
            split_by_paragraphs,
            add_sentence_pauses,
            sentence_pause_ms,
            paragraph_pause_ms,
        ],
    )

    language.change(
        fn=update_voice_choices,
        inputs=[
            language,
            gender,
            voice,
            speed,
            split_by_paragraphs,
            add_sentence_pauses,
            sentence_pause_ms,
            paragraph_pause_ms,
        ],
        outputs=[
            voice,
            settings_state,
        ],
    )

    gender.change(
        fn=update_voice_choices,
        inputs=[
            language,
            gender,
            voice,
            speed,
            split_by_paragraphs,
            add_sentence_pauses,
            sentence_pause_ms,
            paragraph_pause_ms,
        ],
        outputs=[
            voice,
            settings_state,
        ],
    )

    for component in (
        voice,
        speed,
        split_by_paragraphs,
        add_sentence_pauses,
        sentence_pause_ms,
        paragraph_pause_ms,
    ):
        component.change(
            fn=save_settings,
            inputs=[
                language,
                gender,
                voice,
                speed,
                split_by_paragraphs,
                add_sentence_pauses,
                sentence_pause_ms,
                paragraph_pause_ms,
            ],
            outputs=settings_state,
        )

    load_excel_paste_btn.click(
        fn=parse_excel_paste,
        inputs=excel_paste,
        outputs=batch_input,
    )

    preview_selector.change(
        fn=choose_preview_audio,
        inputs=preview_selector,
        outputs=audio_output,
    )

    delete_files_btn.click(
        fn=delete_output_files,
        outputs=[
            audio_output,
            file_output,
            generated_names,
            generation_results,
            preview_selector,
            cleanup_status,
        ],
    )

    generate_btn.click(
        fn=generate_audio,
        inputs=[
            text_input,
            language,
            gender,
            voice,
            speed,
            split_by_paragraphs,
            add_sentence_pauses,
            sentence_pause_ms,
            paragraph_pause_ms,
        ],
        outputs=[
            audio_output,
            file_output,
            generated_names,
            generation_results,
            preview_selector,
            cleanup_status,
        ],
    )

    generate_batch_btn.click(
        fn=generate_batch_audio,
        inputs=[
            batch_input,
            language,
            gender,
            voice,
            speed,
            split_by_paragraphs,
            add_sentence_pauses,
            sentence_pause_ms,
            paragraph_pause_ms,
        ],
        outputs=[
            audio_output,
            file_output,
            generated_names,
            generation_results,
            preview_selector,
            cleanup_status,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        css=APP_CSS,
    )
