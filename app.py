import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths ---
BASE_DIR = Path(__file__).parent
CHECKPOINT = Path.home() / "Documents/projects/igbotts/checkpoints/igbo_tts_f5_wrapped.pt"
VOCAB = Path.home() / "Documents/projects/igbotts/checkpoints/vocab.txt"

REF_AUDIO = {
    "female": {
        "path": str(BASE_DIR / "ref_audio" / "female_1.wav"),
        "text": "Igbe dị na peeji a tụrụ aro ụfọdụ banyere otú ị ga - esi ebido ya.",
    },
    "male": {
        "path": str(BASE_DIR / "ref_audio" / "male_1.wav"),
        "text": "Ọrịa anụahụ nwere ike gbasaa ngwa ngwa n'gburugburu juputara eju.",
    },
}

# --- Device detection ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")

# --- Global singletons ---
f5tts_model = None
translator_model = None
translator_tokenizer = None

# TTS quality/speed: 32=best, 16=good, 8=fast
NFE_STEPS = 16


def load_translator():
    global translator_model, translator_tokenizer
    logger.info("Loading NLLB-200 translation model...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "facebook/nllb-200-distilled-600M"
    translator_tokenizer = AutoTokenizer.from_pretrained(model_name)
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator_model.to(DEVICE if DEVICE != "mps" else "cpu")
    # Warm up with a dummy inference
    inputs = translator_tokenizer("hello", return_tensors="pt", padding=True)
    device = next(translator_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        translator_model.generate(
            **inputs,
            forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids("ibo_Latn"),
            max_new_tokens=8,
        )
    logger.info("NLLB-200 model loaded and warmed up.")


def load_tts():
    global f5tts_model
    logger.info("Loading F5-TTS model...")
    from f5_tts.api import F5TTS

    f5tts_model = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=str(CHECKPOINT),
        vocab_file=str(VOCAB),
        device=DEVICE,
    )
    logger.info("F5-TTS model loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload both models at startup so first request is fast
    load_translator()
    load_tts()
    yield


app = FastAPI(title="SpeakIgbo", lifespan=lifespan)


def _nllb_translate(text: str, device) -> str:
    translator_tokenizer.src_lang = "eng_Latn"
    inputs = translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = translator_model.generate(
            **inputs,
            forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids("ibo_Latn"),
            max_new_tokens=128,
            repetition_penalty=1.2,
        )
    return translator_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


# Common informal contractions NLLB can't parse
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

import re

def _expand_contractions(text: str) -> str:
    """Expand informal contractions so NLLB can parse them."""
    def _replace(m):
        word = m.group(0)
        return _EXPANSIONS.get(word.lower(), word)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in _EXPANSIONS) + r")\b", re.IGNORECASE)
    return pattern.sub(_replace, text)


def _normalize_text(text: str) -> str:
    """Capitalize sentences and add trailing period — NLLB works much better with proper formatting."""
    text = _expand_contractions(text)
    # Split on sentence-ending punctuation, capitalize each sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sentences:
        s = s.strip()
        if s:
            s = s[0].upper() + s[1:]
            result.append(s)
    text = " ".join(result)
    # Add period if no sentence-ending punctuation
    if text and text[-1] not in ".!?":
        text += "."
    return text


def translate_to_igbo(text: str) -> str:
    device = next(translator_model.parameters()).device
    # Always normalize first — NLLB needs proper capitalization and punctuation
    normalized = _normalize_text(text)
    result = _nllb_translate(normalized, device)
    # If NLLB still echoed back, something is really off — return what we got
    return result


class TranslateRequest(BaseModel):
    text: str


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "female"


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    t0 = time.time()
    igbo_text = translate_to_igbo(req.text.strip())
    logger.info(f"Translation: {time.time()-t0:.2f}s — '{req.text.strip()[:50]}' → '{igbo_text[:50]}'")
    return {"igbo_text": igbo_text}


@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    voice = req.voice if req.voice in REF_AUDIO else "female"
    ref = REF_AUDIO[voice]
    igbo_text = req.text.strip()

    t0 = time.time()
    wav, sr, _ = f5tts_model.infer(
        ref_file=ref["path"],
        ref_text=ref["text"],
        gen_text=igbo_text,
        nfe_step=NFE_STEPS,
    )
    t1 = time.time()

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode()
    t2 = time.time()

    logger.info(f"Synthesis: {t1-t0:.2f}s infer + {t2-t1:.2f}s encode ({NFE_STEPS} steps, {len(wav)/sr:.1f}s audio)")
    return {"audio": audio_b64}


# Serve static files (must be after API routes)
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
