from pathlib import Path
from datetime import datetime

import gradio as gr
import numpy as np
import soundfile as sf
from kokoro import KPipeline


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 24000

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


def get_pipeline(language_name: str):
    lang_code = LANGUAGES[language_name]

    if lang_code not in PIPELINES:
        PIPELINES[lang_code] = KPipeline(lang_code=lang_code)

    return PIPELINES[lang_code]


def update_voice_choices(language_name: str):
    voices = VOICES_BY_LANGUAGE.get(language_name, [])
    default_voice = voices[0] if voices else None
    return gr.update(choices=voices, value=default_voice)


def generate_audio(
    text: str,
    language_name: str,
    voice: str,
    speed: float,
    split_by_paragraphs: bool,
):
    if not text or not text.strip():
        raise gr.Error("Please paste some text first.")

    if not voice:
        raise gr.Error("Please select a voice.")

    pipeline = get_pipeline(language_name)

    split_pattern = r"\n+" if split_by_paragraphs else None

    generator = pipeline(
        text.strip(),
        voice=voice,
        speed=speed,
        split_pattern=split_pattern,
    )

    audio_chunks = []

    for index, (graphemes, phonemes, audio) in enumerate(generator):
        audio_chunks.append(audio)

    if not audio_chunks:
        raise gr.Error("No audio was generated. Try shorter text or a different voice.")

    final_audio = np.concatenate(audio_chunks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"kokoro_{language_name.replace(' ', '_')}_{voice}_{timestamp}.wav"

    sf.write(output_path, final_audio, SAMPLE_RATE)

    return str(output_path), str(output_path)


with gr.Blocks(title="Kokoro TTS Studio") as demo:
    gr.Markdown(
        """
        # Kokoro TTS Studio

        Paste text, select language and voice, then generate natural-sounding audio locally.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text",
                placeholder="Paste your narration script here...",
                lines=12,
            )

            with gr.Row():
                language = gr.Dropdown(
                    label="Language",
                    choices=list(LANGUAGES.keys()),
                    value="English - American",
                )

                voice = gr.Dropdown(
                    label="Voice",
                    choices=VOICES_BY_LANGUAGE["English - American"],
                    value="af_heart",
                )

            with gr.Row():
                speed = gr.Slider(
                    label="Speed",
                    minimum=0.75,
                    maximum=1.35,
                    value=1.0,
                    step=0.05,
                )

                split_by_paragraphs = gr.Checkbox(
                    label="Split by paragraphs",
                    value=True,
                )

            generate_btn = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Preview",
                type="filepath",
            )

            file_output = gr.File(
                label="Download WAV",
            )

    language.change(
        fn=update_voice_choices,
        inputs=language,
        outputs=voice,
    )

    generate_btn.click(
        fn=generate_audio,
        inputs=[
            text_input,
            language,
            voice,
            speed,
            split_by_paragraphs,
        ],
        outputs=[
            audio_output,
            file_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )