const textInput = document.getElementById("text-input");
const igboOutput = document.getElementById("igbo-output");
const speakBtn = document.getElementById("speak-btn");
const status = document.getElementById("status");
const audioPlayer = document.getElementById("audio-player");
const gameContainer = document.getElementById("game-container");
const gameCanvas = document.getElementById("game-canvas");

let voice = "female";

// Persistent best score
let bestScore = parseInt(localStorage.getItem("jollof-best") || "0", 10);
const bestEl = document.getElementById("game-best");
if (bestEl) bestEl.textContent = bestScore;

// ============================================================
// Eat the Jollof — mini game while waiting
// ============================================================
const game = (() => {
  let ctx;
  let running = false;
  let animFrame = null;
  let score = 0;
  let speed = 4;
  let frameCount = 0;
  let dpr = 1;
  // Logical (CSS) dimensions — all game math uses these
  let W = 0;
  let H = 0;

  const player = { x: 40, y: 0, w: 26, h: 26, lane: 1 };
  const LANES = 3;
  const LANE_H = 38;
  const TOP_PAD = 8;

  function laneY(lane) {
    return TOP_PAD + lane * LANE_H + (LANE_H - player.h) / 2;
  }

  let items = [];
  // Pre-rendered lane lines (offscreen canvas, drawn once)
  let bgCanvas = null;

  function buildBg() {
    bgCanvas = document.createElement("canvas");
    bgCanvas.width = gameCanvas.width;   // physical pixels
    bgCanvas.height = gameCanvas.height;
    const bg = bgCanvas.getContext("2d");
    bg.scale(dpr, dpr);
    bg.strokeStyle = "rgba(255,255,255,0.08)";
    bg.setLineDash([4, 8]);
    bg.lineWidth = 1;
    for (let i = 1; i < LANES; i++) {
      const ly = TOP_PAD + i * LANE_H;
      bg.beginPath();
      bg.moveTo(0, ly);
      bg.lineTo(W, ly);
      bg.stroke();
    }
  }

  function spawnItem() {
    const lane = Math.floor(Math.random() * LANES);
    const isJollof = Math.random() < 0.65;
    items.push({ x: W + 10, lane, w: 22, h: 22, type: isJollof ? "jollof" : "pepper" });
  }

  function reset() {
    score = 0;
    speed = 4;
    frameCount = 0;
    items = [];
    player.lane = 1;
    player.y = laneY(1);
    updateScore();
  }

  function updateScore() {
    const el = document.getElementById("game-score");
    if (el) el.textContent = score;
    if (score > bestScore) {
      bestScore = score;
      localStorage.setItem("jollof-best", bestScore);
      if (bestEl) bestEl.textContent = bestScore;
    }
  }

  function drawPlayer() {
    const cx = player.x + 13, cy = player.y + 13;
    // Pac-man mouth
    ctx.fillStyle = "#22c55e";
    ctx.beginPath();
    ctx.arc(cx, cy, 13, 0.25, Math.PI * 2 - 0.25);
    ctx.lineTo(cx, cy);
    ctx.fill();
    // Eye
    ctx.fillStyle = "#fff";
    ctx.fillRect(cx + 2, cy - 7, 5, 5);
    ctx.fillStyle = "#000";
    ctx.fillRect(cx + 4, cy - 6, 2.5, 2.5);
  }

  function drawJollof(x, y) {
    // Bowl
    ctx.fillStyle = "#f97316";
    ctx.beginPath();
    ctx.moveTo(x, y + 14);
    ctx.quadraticCurveTo(x + 11, y + 24, x + 22, y + 14);
    ctx.fill();
    // Rice mound
    ctx.fillStyle = "#fb923c";
    ctx.beginPath();
    ctx.arc(x + 11, y + 10, 9, Math.PI, 0);
    ctx.fill();
  }

  function drawPepper(x, y) {
    // Body
    ctx.fillStyle = "#ef4444";
    ctx.beginPath();
    ctx.moveTo(x + 11, y + 2);
    ctx.quadraticCurveTo(x + 22, y + 10, x + 11, y + 22);
    ctx.quadraticCurveTo(x, y + 10, x + 11, y + 2);
    ctx.fill();
    // Stem
    ctx.fillStyle = "#16a34a";
    ctx.fillRect(x + 9, y - 2, 4, 6);
  }

  function collides(px, py, pw, ph, ix, iy, iw, ih) {
    return px < ix + iw && px + pw > ix && py < iy + ih && py + ph > iy;
  }

  let flashAlpha = 0;
  let flashColor = "";

  function tick() {
    if (!running) return;
    frameCount++;

    // Clear + draw cached lane bg
    ctx.clearRect(0, 0, W, H);
    if (bgCanvas) ctx.drawImage(bgCanvas, 0, 0, W, H);

    // Smooth player movement
    const targetY = laneY(player.lane);
    player.y += (targetY - player.y) * 0.3;

    drawPlayer();

    // Spawn
    const spawnRate = Math.max(22, 50 - Math.floor(score / 5) * 2);
    if (frameCount % spawnRate === 0) spawnItem();

    // Update & draw items
    for (let i = items.length - 1; i >= 0; i--) {
      const it = items[i];
      it.x -= speed;
      const iy = laneY(it.lane);

      if (it.type === "jollof") drawJollof(it.x, iy);
      else drawPepper(it.x, iy);

      if (collides(player.x, player.y, player.w, player.h, it.x, iy, it.w, it.h)) {
        if (it.type === "jollof") {
          score++;
          flashColor = "rgba(249,115,22,";
        } else {
          score = Math.max(0, score - 2);
          flashColor = "rgba(239,68,68,";
        }
        flashAlpha = 0.25;
        items.splice(i, 1);
        updateScore();
        continue;
      }

      if (it.x + it.w < 0) items.splice(i, 1);
    }

    // Flash overlay (fades out)
    if (flashAlpha > 0.01) {
      ctx.fillStyle = flashColor + flashAlpha + ")";
      ctx.fillRect(0, 0, W, H);
      flashAlpha *= 0.85;
    }

    speed = 4 + Math.floor(score / 8) * 0.5;
    animFrame = requestAnimationFrame(tick);
  }

  function onKey(e) {
    if (!running) return;
    if (e.key === "ArrowUp" || e.key === "w") {
      e.preventDefault();
      player.lane = Math.max(0, player.lane - 1);
    } else if (e.key === "ArrowDown" || e.key === "s") {
      e.preventDefault();
      player.lane = Math.min(LANES - 1, player.lane + 1);
    }
  }

  let touchStartY = 0;
  function onTouchStart(e) {
    touchStartY = e.touches[0].clientY;
  }
  function onTouchMove(e) {
    if (!running) return;
    const dy = e.touches[0].clientY - touchStartY;
    if (Math.abs(dy) > 12) {
      if (dy < 0) player.lane = Math.max(0, player.lane - 1);
      else player.lane = Math.min(LANES - 1, player.lane + 1);
      touchStartY = e.touches[0].clientY;
    }
  }

  return {
    start() {
      dpr = window.devicePixelRatio || 1;
      W = gameContainer.clientWidth - 20;
      H = TOP_PAD + LANES * LANE_H + 8;
      // Set physical canvas size for sharp rendering
      gameCanvas.width = W * dpr;
      gameCanvas.height = H * dpr;
      // Set CSS display size
      gameCanvas.style.width = W + "px";
      gameCanvas.style.height = H + "px";

      ctx = gameCanvas.getContext("2d");
      ctx.scale(dpr, dpr);

      buildBg();
      reset();
      running = true;
      gameContainer.classList.remove("hidden");
      gameContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
      document.addEventListener("keydown", onKey);
      gameCanvas.addEventListener("touchstart", onTouchStart, { passive: true });
      gameCanvas.addEventListener("touchmove", onTouchMove, { passive: true });
      tick();
    },
    stop() {
      running = false;
      if (animFrame) cancelAnimationFrame(animFrame);
      document.removeEventListener("keydown", onKey);
      gameCanvas.removeEventListener("touchstart", onTouchStart);
      gameCanvas.removeEventListener("touchmove", onTouchMove);
      const finalScore = score;
      gameContainer.classList.add("hidden");
      return finalScore;
    },
  };
})();

// ============================================================
// Toggle wiring
// ============================================================
function setupToggle(id, onChange) {
  const toggle = document.getElementById(id);
  toggle.addEventListener("click", (e) => {
    const btn = e.target.closest(".toggle-btn");
    if (!btn) return;
    toggle.querySelectorAll(".toggle-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    onChange(btn.dataset.value);
  });
}

setupToggle("voice-toggle", (val) => (voice = val));

// ============================================================
// Speak flow
// ============================================================
speakBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) return;

  speakBtn.disabled = true;
  audioPlayer.classList.add("hidden");
  status.className = "status";
  status.textContent = "";

  // Determine what to synthesize
  let igboText = text;
  // Also check if user manually typed Igbo in the right box
  const manualIgbo = igboOutput.textContent.trim();

  try {
    // Step 1: Translate English → Igbo
    igboOutput.textContent = "Translating…";
    igboOutput.classList.add("translating");

    const tRes = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!tRes.ok) {
      const err = await tRes.json().catch(() => ({ detail: "Translation failed" }));
      throw new Error(err.detail || "Translation failed");
    }

    const tData = await tRes.json();
    igboText = tData.igbo_text;
    igboOutput.textContent = igboText;
    igboOutput.classList.remove("translating");

    // Step 2: Synthesize — start game immediately
    status.innerHTML = '<div class="loading-wave"><span></span><span></span><span></span><span></span><span></span></div><div class="loading-msg">Generating speech…</div>';
    status.className = "status loading-active";

    game.start();

    const sRes = await fetch("/api/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: igboText, voice }),
    });

    if (!sRes.ok) {
      const err = await sRes.json().catch(() => ({ detail: "Synthesis failed" }));
      throw new Error(err.detail || "Synthesis failed");
    }

    const sData = await sRes.json();
    const finalScore = game.stop();

    const audioBytes = Uint8Array.from(atob(sData.audio), (c) => c.charCodeAt(0));
    const blob = new Blob([audioBytes], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    audioPlayer.src = url;
    audioPlayer.classList.remove("hidden");
    audioPlayer.play();

    status.className = "status";
    status.textContent = finalScore > 0 ? `You ate ${finalScore} jollof!` : "";
  } catch (err) {
    game.stop();
    igboOutput.classList.remove("translating");
    status.className = "status error";
    status.textContent = err.message;
  } finally {
    speakBtn.disabled = false;
  }
});

// Ctrl/Cmd+Enter to speak
textInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    speakBtn.click();
  }
});
