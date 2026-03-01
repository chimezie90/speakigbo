"""SpeakIgbo — Modal deployment with split CPU/GPU inference.

CPU container: serves static files + translation (NLLB-200) — fast cold start ~10s
GPU container: TTS synthesis (F5-TTS) — cold start hidden behind game time
"""

import modal
from pathlib import Path

app = modal.App("speakigbo")

# Model volume (persists across deploys)
volume = modal.Volume.from_name("speakigbo-models", create_if_missing=True)

# --- Images ---

# Lightweight CPU image for translation + static serving
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.110.0",
        "transformers",
        "sentencepiece",
        "torch",
    )
    .add_local_dir(Path(__file__).parent / "static", remote_path="/app/static")
)

# GPU image for TTS synthesis
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg", "git")
    .pip_install(
        "f5-tts",
        "soundfile",
        "torch",
        "torchaudio",
    )
    .add_local_dir(Path(__file__).parent / "ref_audio", remote_path="/app/ref_audio")
)


# --- GPU class: TTS synthesis with persistent model loading ---

@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={"/models": volume},
    timeout=120,
    container_idle_timeout=300,
)
@modal.concurrent(max_inputs=4)
class TTSModel:
    @modal.enter()
    def load_model(self):
        import logging
        import torch

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nfe_steps = 16 if self.device == "cuda" else 8
        self.logger.info(f"Loading F5-TTS on {self.device}...")

        from f5_tts.api import F5TTS
        self.f5tts = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file="/models/igbo_tts_f5_wrapped.pt",
            vocab_file="/models/vocab.txt",
            device=self.device,
        )
        self.logger.info("F5-TTS ready.")

        self.ref_audio = {
            "female": {
                "path": "/app/ref_audio/female_1.wav",
                "text": "Igbe dị na peeji a tụrụ aro ụfọdụ banyere otú ị ga - esi ebido ya.",
            },
            "male": {
                "path": "/app/ref_audio/male_1.wav",
                "text": "Ọrịa anụahụ nwere ike gbasaa ngwa ngwa n'gburugburu juputara eju.",
            },
        }

    @modal.method()
    def synthesize(self, igbo_text: str, voice: str = "female") -> dict:
        import base64
        import io
        import time
        import soundfile as sf

        ref = self.ref_audio.get(voice, self.ref_audio["female"])

        t0 = time.time()
        wav, sr, _ = self.f5tts.infer(
            ref_file=ref["path"], ref_text=ref["text"],
            gen_text=igbo_text, nfe_step=self.nfe_steps,
        )
        t1 = time.time()

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()

        self.logger.info(f"Synthesis: {t1-t0:.2f}s ({self.nfe_steps} steps, {len(wav)/sr:.1f}s audio)")
        return {"audio": audio_b64}


# --- CPU function: web server with translation + static files ---

@app.function(
    image=cpu_image,
    timeout=120,
    container_idle_timeout=300,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    import logging
    import re
    import time

    import torch
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load NLLB-200 on CPU — fast and lightweight
    logger.info("Loading NLLB-200 on CPU...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    # Warm up
    _inp = nllb_tokenizer("hello", return_tensors="pt", padding=True)
    with torch.no_grad():
        nllb_model.generate(**_inp, forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("ibo_Latn"), max_new_tokens=8)
    logger.info("NLLB-200 ready.")

    # --- Translation helpers ---

    _EXPANSIONS = {
        "hows": "how is", "whats": "what is", "thats": "that is",
        "whos": "who is", "wheres": "where is", "heres": "here is",
        "theres": "there is", "lets": "let us", "dont": "do not",
        "doesnt": "does not", "didnt": "did not", "cant": "can not",
        "wont": "will not", "isnt": "is not", "arent": "are not",
        "wasnt": "was not", "werent": "were not", "im": "I am",
        "ive": "I have", "youre": "you are", "youve": "you have",
        "youll": "you will", "theyre": "they are", "weve": "we have",
        "gonna": "going to", "wanna": "want to", "gotta": "got to",
    }
    _EXPAND_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _EXPANSIONS) + r")\b", re.IGNORECASE)

    def _expand_contractions(text):
        return _EXPAND_RE.sub(lambda m: _EXPANSIONS.get(m.group(0).lower(), m.group(0)), text)

    def _normalize_text(text):
        text = _expand_contractions(text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        for s in sentences:
            s = s.strip()
            if s:
                s = s[0].upper() + s[1:]
                result.append(s)
        text = " ".join(result)
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _nllb_translate(text):
        nllb_tokenizer.src_lang = "eng_Latn"
        inputs = nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            generated = nllb_model.generate(
                **inputs,
                forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("ibo_Latn"),
                max_new_tokens=128,
                repetition_penalty=1.2,
            )
        return nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def translate_to_igbo(text: str) -> str:
        normalized = _normalize_text(text)
        return _nllb_translate(normalized)

    # Reference to GPU TTS class
    tts_model = TTSModel()

    # --- FastAPI ---
    api = FastAPI(title="SpeakIgbo")

    class TranslateRequest(BaseModel):
        text: str

    class SynthesizeRequest(BaseModel):
        text: str
        voice: str = "female"

    @api.post("/api/translate")
    async def translate(req: TranslateRequest):
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        t0 = time.time()
        igbo_text = translate_to_igbo(req.text.strip())
        logger.info(f"Translation: {time.time()-t0:.2f}s")
        return {"igbo_text": igbo_text}

    @api.post("/api/synthesize")
    async def synthesize(req: SynthesizeRequest):
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        voice = req.voice if req.voice in ("female", "male") else "female"
        # Call the GPU function remotely
        result = tts_model.synthesize.remote(req.text.strip(), voice)
        return result

    api.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
    return api


@app.local_entrypoint()
def upload_models():
    """Upload model files to the Modal volume. Run once with: modal run modal_app.py"""
    import os
    home = os.path.expanduser("~")
    checkpoint = os.path.join(home, "Documents/projects/igbotts/checkpoints/igbo_tts_f5_wrapped.pt")
    vocab = os.path.join(home, "Documents/projects/igbotts/checkpoints/vocab.txt")

    entries = list(volume.listdir("/"))
    existing = {e.path for e in entries}
    print(f"Existing files in volume: {existing}")

    if "igbo_tts_f5_wrapped.pt" in existing and "vocab.txt" in existing:
        print("Models already uploaded! Deploy with: modal deploy modal_app.py")
        return

    print(f"Uploading checkpoint ({os.path.getsize(checkpoint)//1024//1024}MB) and vocab to Modal volume...")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(checkpoint, "igbo_tts_f5_wrapped.pt")
        batch.put_file(vocab, "vocab.txt")
    print("Done! Now deploy with: modal deploy modal_app.py")
