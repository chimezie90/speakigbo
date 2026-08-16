"""SpeakIgbo — Modal deployment with split CPU/GPU inference.

CPU container: serves static files + translation (NLLB-200) — fast cold start ~10s
GPU container: TTS synthesis (F5-TTS) — cold start hidden behind game time
"""

import modal
from pathlib import Path

app = modal.App("speakigbo")

# Model volume (persists across deploys)
volume = modal.Volume.from_name("speakigbo-models", create_if_missing=True)

# Distributed counters for beta rate-limiting. Survives across containers and
# cold starts, unlike per-process memory — this is what bounds GPU cost.
rate_dict = modal.Dict.from_name("speakigbo-ratelimit", create_if_missing=True)

# Anonymous usage counters (telemetry) and the opt-in marketing list.
events_dict = modal.Dict.from_name("speakigbo-events", create_if_missing=True)
subscribers_dict = modal.Dict.from_name("speakigbo-subscribers", create_if_missing=True)

# Beta usage caps — tune freely.
PER_IP_DAILY_CALLS = 150    # translate + synth calls per client IP per day
GLOBAL_DAILY_SYNTH = 3000   # hard ceiling on GPU synth calls/day (cost breaker)
MAX_SYNTH_CHARS = 500       # per synth request — bounds GPU-seconds per call
MAX_TRANSLATE_CHARS = 3000  # per translate request
MAX_SENTENCES = 40          # cap sentences translated per request (CPU bound)
MAX_CONTACT_CHARS = 200     # email/phone length cap on the subscribe list

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
    .run_commands("echo 'v6-faster-female'")
    .add_local_dir(Path(__file__).parent / "static", remote_path="/app/static")
)

# GPU image for TTS synthesis + ASR
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg", "git")
    .pip_install(
        "f5-tts",
        "soundfile",
        "torch",
        "torchaudio",
        "transformers",
        "safetensors",
        "pyctcdecode",
        "kenlm",
        "librosa",
    )
    .add_local_dir(Path(__file__).parent / "ref_audio", remote_path="/app/ref_audio")
)


# --- GPU class: TTS synthesis with persistent model loading ---

@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={"/models": volume},
    timeout=120,
    scaledown_window=300,
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

        # --- ASR model (MMS-1B + v7 adapter) ---
        self.logger.info("Loading ASR model (MMS-1B + v7 adapter)...")
        from safetensors.torch import load_file as safe_load_file
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE

        asr_dir = Path("/models/asr_v7")
        self.asr_processor = Wav2Vec2Processor.from_pretrained(str(asr_dir))
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/mms-1b-all",
            vocab_size=self.asr_processor.tokenizer.vocab_size,
            ignore_mismatched_sizes=True,
        )
        adapter_file = asr_dir / WAV2VEC2_ADAPTER_SAFE_FILE.format("ibo")
        adapter_weights = safe_load_file(str(adapter_file))
        model_state = self.asr_model.state_dict()
        model_state.update(adapter_weights)
        self.asr_model.load_state_dict(model_state)
        self.asr_model.eval()
        self.asr_model.to(self.device)
        self.logger.info("ASR model loaded.")

        # KenLM beam decoder
        self.asr_beam_decoder = None
        kenlm_path = Path("/models/kenlm/igbo_5gram.bin")
        if kenlm_path.exists():
            try:
                from pyctcdecode import build_ctcdecoder
                vocab = self.asr_processor.tokenizer.get_vocab()
                sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
                labels = [token for token, _ in sorted_vocab]
                model_vocab_size = self.asr_model.config.vocab_size
                if len(labels) > model_vocab_size:
                    labels = labels[:model_vocab_size]
                self.asr_beam_decoder = build_ctcdecoder(
                    labels=labels,
                    kenlm_model_path=str(kenlm_path),
                    alpha=0.5,
                    beta=1.5,
                )
                self.logger.info("KenLM beam decoder loaded.")
            except Exception as e:
                self.logger.warning(f"Failed to build beam decoder: {e}")
        else:
            self.logger.info("KenLM not found, using greedy decoding.")

        self.ref_audio = {
            "female": {
                "path": "/app/ref_audio/female_1.wav",
                "text": "O doro anya na nke a bụ ihe na-adọrọ mmasị ọtụtụ n'ịme emume Eucharist.",
            },
            "male": {
                "path": "/app/ref_audio/male_1.wav",
                "text": "Ọbịbịa ya na-agwaghị onye ọbụla wetara ajụjụ karịa ọsịsa na njikọ ha nke dịkwa mfe ịgbaji.",
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

    @modal.method()
    def warmup(self) -> dict:
        # No inference. @modal.enter already loaded the model when the container
        # started; invoking this via .spawn() just ensures a warm container
        # exists so the next real synth skips cold start.
        return {"ready": True}

    @modal.method()
    def transcribe(self, audio_b64: str) -> dict:
        import base64
        import io
        import time

        import librosa
        import numpy as np
        import soundfile as sf
        import torch

        t0 = time.time()

        audio_bytes = base64.b64decode(audio_b64)
        audio_arr, sr = sf.read(io.BytesIO(audio_bytes))

        # Convert stereo to mono
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=1)

        # Resample to 16kHz
        if sr != 16000:
            audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
            sr = 16000

        duration = len(audio_arr) / sr

        inputs = self.asr_processor(audio_arr, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.asr_model(**inputs).logits

        if self.asr_beam_decoder is not None:
            logits_np = logits.cpu().numpy()[0]
            text = self.asr_beam_decoder.decode(logits_np, beam_width=100)
        else:
            pred_ids = torch.argmax(logits, dim=-1)
            text = self.asr_processor.batch_decode(pred_ids)[0]

        elapsed = time.time() - t0
        self.logger.info(f"Transcribed {duration:.1f}s audio in {elapsed:.1f}s: {text[:80]!r}")

        return {
            "text": text,
            "duration_seconds": round(duration, 2),
            "inference_time": round(elapsed, 2),
        }


# --- CPU function: web server with translation + static files ---

@app.function(
    image=cpu_image,
    timeout=120,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("speakigbo-beta")],
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    import datetime
    import logging
    import os
    import re
    import time

    import torch
    from fastapi import FastAPI, HTTPException, Request
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

    def _dedup_words(text):
        """Collapse runs of 3+ identical words — the NLLB degenerate-loop artifact
        (e.g. 'dozi dozi dozi dozi'). A 2x pair is left intact because Igbo
        reduplication is grammatical ('obere obere' = little by little, 'ọsọ ọsọ'
        = quickly), so we must not strip it."""
        words = text.split()
        out = []
        run = 0
        for w in words:
            if out and out[-1].lower() == w.lower():
                run += 1
                if run >= 2:  # 3rd+ identical in a row → drop
                    continue
            else:
                run = 0
            out.append(w)
        return " ".join(out)

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
        result = nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return _dedup_words(result)

    def _split_sentences(text):
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def translate_to_igbo(text: str) -> str:
        normalized = _normalize_text(text)
        sentences = _split_sentences(normalized)[:MAX_SENTENCES]
        if len(sentences) <= 1:
            return _nllb_translate(normalized)
        return " ".join(_nllb_translate(s) for s in sentences)

    # Reference to GPU TTS class
    tts_model = TTSModel()

    # --- Beta gating: access key + rate limits ---
    BETA_KEY = os.environ.get("BETA_KEY", "")

    def _today() -> str:
        # UTC so the daily reset is container-timezone-independent.
        return datetime.datetime.now(datetime.timezone.utc).date().isoformat()

    def _client_ip(request: Request) -> str:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            # Trust only the right-most entry (the value Modal's proxy appends).
            # Left-most entries are client-supplied and spoofable, which would
            # defeat the per-IP cap.
            return xff.split(",")[-1].strip()
        return request.client.host if request.client else "unknown"

    def _gate(request: Request, gpu: bool = False):
        # 1. Beta access key. Fail CLOSED — if the key isn't configured, refuse
        #    rather than silently exposing the GPU endpoints to the open internet.
        if not BETA_KEY:
            raise HTTPException(status_code=503, detail="Service unavailable.")
        if request.headers.get("x-beta-key") != BETA_KEY:
            raise HTTPException(status_code=401, detail="Beta access key required.")
        today = _today()
        # 2. Per-IP daily cap
        ip_key = f"{today}:ip:{_client_ip(request)}"
        ip_n = rate_dict.get(ip_key, 0)
        if ip_n >= PER_IP_DAILY_CALLS:
            raise HTTPException(status_code=429, detail="Daily limit reached for this device. Try again tomorrow.")
        rate_dict[ip_key] = ip_n + 1
        # 3. Global GPU synth budget — cost circuit-breaker. Best-effort under
        #    concurrency (Dict increments aren't atomic); the per-request length
        #    cap below is what actually bounds cost per call.
        if gpu:
            g_key = f"{today}:global_synth"
            g_n = rate_dict.get(g_key, 0)
            if g_n >= GLOBAL_DAILY_SYNTH:
                raise HTTPException(status_code=429, detail="Beta is at capacity today — please try again later.")
            rate_dict[g_key] = g_n + 1

    # --- FastAPI ---
    api = FastAPI(title="SpeakIgbo")

    # goodbye.html (uninstall survey) posts to /api/event from the vercel site;
    # browser pages need CORS, unlike the extensions (host_permissions bypass it).
    from fastapi.middleware.cors import CORSMiddleware
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["https://igbo-keyboard.vercel.app"],
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )

    class TranslateRequest(BaseModel):
        text: str

    class SynthesizeRequest(BaseModel):
        text: str
        voice: str = "female"

    class TranscribeRequest(BaseModel):
        audio: str  # base64-encoded audio

    class TranslateToEnglishRequest(BaseModel):
        text: str

    class EventRequest(BaseModel):
        event: str
        clientId: str = ""
        product: str = ""
        ts: int = 0

    class SubscribeRequest(BaseModel):
        email: str = ""
        phone: str = ""
        product: str = ""
        source: str = ""

    def _light_limit(request: Request, bucket: str, cap: int):
        """Per-IP daily cap for the open (no-key) endpoints — cheap abuse guard."""
        today = _today()
        k = f"lim:{today}:{bucket}:{_client_ip(request)}"
        n = events_dict.get(k, 0)
        if n >= cap:
            raise HTTPException(status_code=429, detail="Too many requests today.")
        events_dict[k] = n + 1

    @api.post("/api/translate")
    async def translate(req: TranslateRequest, request: Request):
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if len(req.text) > MAX_TRANSLATE_CHARS:
            raise HTTPException(status_code=400, detail=f"Text too long (max {MAX_TRANSLATE_CHARS} chars).")
        _gate(request)
        t0 = time.time()
        igbo_text = translate_to_igbo(req.text.strip())
        logger.info(f"Translation: {time.time()-t0:.2f}s")
        return {"igbo_text": igbo_text}

    @api.post("/api/synthesize")
    async def synthesize(req: SynthesizeRequest, request: Request):
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if len(req.text) > MAX_SYNTH_CHARS:
            raise HTTPException(status_code=400, detail=f"Text too long (max {MAX_SYNTH_CHARS} chars).")
        _gate(request, gpu=True)
        voice = req.voice if req.voice in ("female", "male") else "female"
        # Call the GPU function remotely
        result = tts_model.synthesize.remote(req.text.strip(), voice)
        return result

    @api.post("/api/transcribe")
    async def transcribe_audio(req: TranscribeRequest, request: Request):
        if not req.audio:
            raise HTTPException(status_code=400, detail="Audio data is required")
        _gate(request, gpu=True)
        result = tts_model.transcribe.remote(req.audio)
        return result

    @api.post("/api/translate-to-english")
    async def translate_to_english(req: TranslateToEnglishRequest, request: Request):
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if len(req.text) > MAX_TRANSLATE_CHARS:
            raise HTTPException(status_code=400, detail=f"Text too long (max {MAX_TRANSLATE_CHARS} chars).")
        _gate(request)
        t0 = time.time()
        nllb_tokenizer.src_lang = "ibo_Latn"
        inputs = nllb_tokenizer(req.text.strip(), return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            generated = nllb_model.generate(
                **inputs,
                forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=128,
                repetition_penalty=1.2,
            )
        english_text = nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        logger.info(f"Igbo→English: {time.time()-t0:.2f}s")
        return {"english_text": english_text}

    @api.post("/api/event")
    async def track_event(req: EventRequest, request: Request):
        # Open (no beta key) — the public keyboard reports here too. Anonymous:
        # event name + counts, never text. Light per-IP cap guards against spam.
        _light_limit(request, "event", 2000)
        ev = (req.event or "unknown")[:40]
        product = (req.product or "unknown")[:20]
        today = _today()
        for k in (f"ct:{product}:{ev}", f"ct:{today}:{product}:{ev}"):
            events_dict[k] = events_dict.get(k, 0) + 1
        # Daily active clients (dedup by anonymous clientId)
        if req.clientId:
            seen = f"seen:{today}:{product}:{req.clientId[:64]}"
            if seen not in events_dict:
                events_dict[seen] = 1
                dau = f"dau:{today}:{product}"
                events_dict[dau] = events_dict.get(dau, 0) + 1
        return {"ok": True}

    @api.post("/api/warmup")
    async def warmup_route(request: Request):
        # Spins up / keeps a GPU container warm WITHOUT running a synth, so the
        # first real read skips cold start. Fire-and-forget; does not count
        # against the synth budget.
        _light_limit(request, "warmup", 200)
        tts_model.warmup.spawn()
        return {"ok": True}

    @api.post("/api/subscribe")
    async def subscribe(req: SubscribeRequest, request: Request):
        # Opt-in marketing list. Open endpoint, tight per-IP cap.
        _light_limit(request, "sub", 30)
        email = (req.email or "").strip().lower()[:MAX_CONTACT_CHARS]
        phone = (req.phone or "").strip()[:MAX_CONTACT_CHARS]
        if not email and not phone:
            raise HTTPException(status_code=400, detail="Email or phone required.")
        if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            raise HTTPException(status_code=400, detail="Invalid email address.")
        if phone and not re.match(r"^[+0-9][0-9 ()-]{5,}$", phone):
            raise HTTPException(status_code=400, detail="Invalid phone number.")
        key = f"sub:{email or phone}"
        existing = subscribers_dict.get(key)
        subscribers_dict[key] = {
            "email": email,
            "phone": phone,
            "product": (req.product or "")[:20],
            "source": (req.source or "")[:40],
            "ts": existing.get("ts") if existing else time.time(),
        }
        return {"ok": True}

    api.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
    return api


@app.local_entrypoint()
def export_subscribers():
    """Export the opt-in marketing list to CSV. Run: modal run modal_app.py::export_subscribers"""
    import csv

    def _safe(val):
        # Neutralize CSV formula injection — a cell starting with = + - @ can
        # execute when the exported file is opened in Excel/Sheets.
        s = str(val)
        return "'" + s if s[:1] in ("=", "+", "-", "@") else s

    subs = modal.Dict.from_name("speakigbo-subscribers", create_if_missing=True)
    rows = [v for _, v in subs.items()]
    rows.sort(key=lambda r: r.get("ts", 0))
    out = "subscribers.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["email", "phone", "product", "source", "ts"])
        w.writeheader()
        for r in rows:
            w.writerow({k: _safe(r.get(k, "")) for k in w.fieldnames})
    print(f"Wrote {len(rows)} subscribers to {out}")


@app.local_entrypoint()
def stats():
    """Print usage counters. Run: modal run modal_app.py::stats"""
    events = modal.Dict.from_name("speakigbo-events", create_if_missing=True)
    interesting = sorted(
        (k, v) for k, v in events.items()
        if k.startswith("ct:") or k.startswith("dau:")
    )
    for k, v in interesting:
        print(f"{k}\t{v}")


@app.local_entrypoint()
def upload_models():
    """Upload model files to the Modal volume. Run once with: modal run modal_app.py"""
    import os
    home = os.path.expanduser("~")
    igbotts = os.path.join(home, "Documents/projects/igbotts")

    # TTS files
    checkpoint = os.path.join(igbotts, "checkpoints/igbo_tts_f5_wrapped.pt")
    vocab = os.path.join(igbotts, "checkpoints/vocab.txt")

    # ASR files
    asr_dir = os.path.join(igbotts, "asr_checkpoints/v7")
    kenlm_file = os.path.join(igbotts, "asr_checkpoints/kenlm/igbo_5gram.bin")

    # Check what's already on the volume
    entries = list(volume.listdir("/"))
    existing = {e.path for e in entries}
    print(f"Existing files in volume: {existing}")

    print("Uploading models to Modal volume...")
    with volume.batch_upload(force=True) as batch:
        # TTS (skip if already uploaded)
        if os.path.isfile(checkpoint):
            batch.put_file(checkpoint, "igbo_tts_f5_wrapped.pt")
            print(f"  igbo_tts_f5_wrapped.pt ({os.path.getsize(checkpoint)//1024//1024}MB)")
        elif "igbo_tts_f5_wrapped.pt" not in existing:
            print("  WARNING: TTS checkpoint not found locally and not on volume!")
        else:
            print("  TTS checkpoint already on volume, skipping.")

        if os.path.isfile(vocab):
            batch.put_file(vocab, "vocab.txt")
        elif "vocab.txt" not in existing:
            print("  WARNING: vocab.txt not found locally and not on volume!")

        # ASR v7 adapter
        for fname in os.listdir(asr_dir):
            fpath = os.path.join(asr_dir, fname)
            if os.path.isfile(fpath):
                batch.put_file(fpath, f"asr_v7/{fname}")
                print(f"  asr_v7/{fname} ({os.path.getsize(fpath)//1024}KB)")

        # KenLM
        batch.put_file(kenlm_file, "kenlm/igbo_5gram.bin")
        print(f"  kenlm/igbo_5gram.bin ({os.path.getsize(kenlm_file)//1024//1024}MB)")
    print("Done! Now deploy with: modal deploy modal_app.py")
