"""SpeakIgbo — Modal deployment with GPU inference."""

import modal
from pathlib import Path

app = modal.App("speakigbo")

# Model volume (persists across deploys)
volume = modal.Volume.from_name("speakigbo-models", create_if_missing=True)

# Docker image with deps + local static/ref_audio baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg", "git")
    .pip_install(
        "fastapi>=0.110.0",
        "f5-tts",
        "transformers",
        "sentencepiece",
        "soundfile",
        "torch",
        "torchaudio",
    )
    .add_local_dir(Path(__file__).parent / "static", remote_path="/app/static")
    .add_local_dir(Path(__file__).parent / "ref_audio", remote_path="/app/ref_audio")
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": volume},
    timeout=120,
    container_idle_timeout=300,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import base64
    import io
    import logging
    import time

    import soundfile as sf
    import torch
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")
    NFE_STEPS = 16 if DEVICE == "cuda" else 8

    CHECKPOINT = "/models/igbo_tts_f5_wrapped.pt"
    VOCAB = "/models/vocab.txt"

    REF_AUDIO = {
        "female": {
            "path": "/app/ref_audio/female_1.wav",
            "text": "Igbe dị na peeji a tụrụ aro ụfọdụ banyere otú ị ga - esi ebido ya.",
        },
        "male": {
            "path": "/app/ref_audio/male_1.wav",
            "text": "Ọrịa anụahụ nwere ike gbasaa ngwa ngwa n'gburugburu juputara eju.",
        },
    }

    # Load models eagerly on container start
    logger.info("Loading NLLB-200...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(DEVICE)
    _inp = nllb_tokenizer("hello", return_tensors="pt", padding=True)
    _inp = {k: v.to(DEVICE) for k, v in _inp.items()}
    with torch.no_grad():
        nllb_model.generate(**_inp, forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("ibo_Latn"), max_new_tokens=8)
    logger.info("NLLB-200 ready.")

    logger.info("Loading F5-TTS...")
    from f5_tts.api import F5TTS
    f5tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=CHECKPOINT, vocab_file=VOCAB, device=DEVICE)
    logger.info("F5-TTS ready.")

    # --- FastAPI ---
    api = FastAPI(title="SpeakIgbo")

    import re

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
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
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
        voice = req.voice if req.voice in REF_AUDIO else "female"
        ref = REF_AUDIO[voice]
        igbo_text = req.text.strip()

        t0 = time.time()
        wav, sr, _ = f5tts.infer(
            ref_file=ref["path"], ref_text=ref["text"],
            gen_text=igbo_text, nfe_step=NFE_STEPS,
        )
        t1 = time.time()
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()
        logger.info(f"Synthesis: {t1-t0:.2f}s ({NFE_STEPS} steps, {len(wav)/sr:.1f}s audio)")
        return {"audio": audio_b64}

    api.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
    return api


@app.local_entrypoint()
def upload_models():
    """Upload model files to the Modal volume. Run once with: modal run modal_app.py"""
    import os
    home = os.path.expanduser("~")
    checkpoint = os.path.join(home, "Documents/projects/igbotts/checkpoints/igbo_tts_f5_wrapped.pt")
    vocab = os.path.join(home, "Documents/projects/igbotts/checkpoints/vocab.txt")

    # Check what's already there
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
