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
// Player at bottom, items fall from top. Move left/right.
// ============================================================
const game = (() => {
  let ctx;
  let running = false;
  let animFrame = null;
  let score = 0;
  let fallSpeed = 2.5;
  let frameCount = 0;
  let dpr = 1;
  let W = 0;
  let H = 0;

  const PLAYER_W = 32;
  const PLAYER_H = 28;
  const MOVE_SPEED = 6;
  const player = { x: 0, targetX: 0 };
  const PLAYER_Y_OFFSET = 10; // from bottom

  let items = [];
  let flashAlpha = 0;
  let flashColor = "";

  function playerY() { return H - PLAYER_H - PLAYER_Y_OFFSET; }

  function reset() {
    score = 0;
    fallSpeed = 2.5;
    frameCount = 0;
    items = [];
    player.x = W / 2 - PLAYER_W / 2;
    player.targetX = player.x;
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

  function spawnItem() {
    const size = 22;
    const x = Math.random() * (W - size);
    const isJollof = Math.random() < 0.6;
    items.push({ x, y: -size, w: size, h: size, type: isJollof ? "jollof" : "pepper" });
  }

  function drawPlayer() {
    const px = player.x;
    const py = playerY();
    const cx = px + PLAYER_W / 2;
    const cy = py + PLAYER_H / 2;
    // Open mouth facing up
    ctx.fillStyle = "#22c55e";
    ctx.beginPath();
    ctx.arc(cx, cy, PLAYER_W / 2, -Math.PI * 0.75, -Math.PI * 0.25, true);
    ctx.lineTo(cx, cy);
    ctx.fill();
    // Eye
    ctx.fillStyle = "#fff";
    ctx.fillRect(cx - 3, cy - 8, 5, 5);
    ctx.fillStyle = "#000";
    ctx.fillRect(cx - 2, cy - 7, 2.5, 2.5);
  }

  function drawJollof(x, y) {
    ctx.fillStyle = "#f97316";
    ctx.beginPath();
    ctx.moveTo(x, y + 14);
    ctx.quadraticCurveTo(x + 11, y + 24, x + 22, y + 14);
    ctx.fill();
    ctx.fillStyle = "#fb923c";
    ctx.beginPath();
    ctx.arc(x + 11, y + 10, 9, Math.PI, 0);
    ctx.fill();
  }

  function drawPepper(x, y) {
    ctx.fillStyle = "#ef4444";
    ctx.beginPath();
    ctx.moveTo(x + 11, y + 2);
    ctx.quadraticCurveTo(x + 22, y + 10, x + 11, y + 22);
    ctx.quadraticCurveTo(x, y + 10, x + 11, y + 2);
    ctx.fill();
    ctx.fillStyle = "#16a34a";
    ctx.fillRect(x + 9, y - 2, 4, 6);
  }

  function collides(ax, ay, aw, ah, bx, by, bw, bh) {
    return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
  }

  // Input state
  let moveDir = 0; // -1 left, 0 none, 1 right
  let keysDown = new Set();

  function onKeyDown(e) {
    if (!running) return;
    if (e.key === "ArrowLeft" || e.key === "a") { e.preventDefault(); keysDown.add("left"); }
    if (e.key === "ArrowRight" || e.key === "d") { e.preventDefault(); keysDown.add("right"); }
    updateMoveDir();
  }
  function onKeyUp(e) {
    if (e.key === "ArrowLeft" || e.key === "a") keysDown.delete("left");
    if (e.key === "ArrowRight" || e.key === "d") keysDown.delete("right");
    updateMoveDir();
  }
  function updateMoveDir() {
    if (keysDown.has("left") && !keysDown.has("right")) moveDir = -1;
    else if (keysDown.has("right") && !keysDown.has("left")) moveDir = 1;
    else moveDir = 0;
  }

  // Touch: tap left/right side of canvas to move, hold to keep moving
  let touchHoldInterval = null;
  function onTouchStart(e) {
    if (!running) return;
    e.preventDefault();
    const rect = gameCanvas.getBoundingClientRect();
    const touchX = e.touches[0].clientX - rect.left;
    moveDir = touchX < rect.width / 2 ? -1 : 1;
  }
  function onTouchMove(e) {
    if (!running) return;
    const rect = gameCanvas.getBoundingClientRect();
    const touchX = e.touches[0].clientX - rect.left;
    moveDir = touchX < rect.width / 2 ? -1 : 1;
  }
  function onTouchEnd() {
    moveDir = 0;
  }

  function tick() {
    if (!running) return;
    frameCount++;
    ctx.clearRect(0, 0, W, H);

    // Move player
    player.x += moveDir * MOVE_SPEED;
    player.x = Math.max(0, Math.min(W - PLAYER_W, player.x));

    drawPlayer();

    // Spawn
    const spawnRate = Math.max(18, 45 - Math.floor(score / 5) * 2);
    if (frameCount % spawnRate === 0) spawnItem();

    // Update & draw items
    const py = playerY();
    for (let i = items.length - 1; i >= 0; i--) {
      const it = items[i];
      it.y += fallSpeed;

      if (it.type === "jollof") drawJollof(it.x, it.y);
      else drawPepper(it.x, it.y);

      if (collides(player.x, py, PLAYER_W, PLAYER_H, it.x, it.y, it.w, it.h)) {
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

      // Remove if off screen (missed jollof doesn't penalize)
      if (it.y > H + 10) items.splice(i, 1);
    }

    // Flash overlay
    if (flashAlpha > 0.01) {
      ctx.fillStyle = flashColor + flashAlpha + ")";
      ctx.fillRect(0, 0, W, H);
      flashAlpha *= 0.85;
    }

    fallSpeed = 2.5 + Math.floor(score / 6) * 0.4;
    animFrame = requestAnimationFrame(tick);
  }

  return {
    start() {
      // Unhide FIRST so we can measure dimensions
      gameContainer.classList.remove("hidden");

      dpr = window.devicePixelRatio || 1;
      W = gameContainer.clientWidth - 20;
      H = 150;
      gameCanvas.width = W * dpr;
      gameCanvas.height = H * dpr;
      gameCanvas.style.width = W + "px";
      gameCanvas.style.height = H + "px";

      ctx = gameCanvas.getContext("2d");
      ctx.scale(dpr, dpr);

      reset();
      running = true;
      moveDir = 0;
      keysDown.clear();
      gameContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
      document.addEventListener("keydown", onKeyDown);
      document.addEventListener("keyup", onKeyUp);
      gameCanvas.addEventListener("touchstart", onTouchStart, { passive: false });
      gameCanvas.addEventListener("touchmove", onTouchMove, { passive: false });
      gameCanvas.addEventListener("touchend", onTouchEnd, { passive: true });
      tick();
    },
    stop() {
      running = false;
      if (animFrame) cancelAnimationFrame(animFrame);
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("keyup", onKeyUp);
      gameCanvas.removeEventListener("touchstart", onTouchStart);
      gameCanvas.removeEventListener("touchmove", onTouchMove);
      gameCanvas.removeEventListener("touchend", onTouchEnd);
      moveDir = 0;
      keysDown.clear();
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
