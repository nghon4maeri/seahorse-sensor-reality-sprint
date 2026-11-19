/**
 * Accessible Before/After Split-Screen Image Comparator (WCAG 2.2 AA).
 * Supports pointer drag, touch drag, keyboard arrows/Home/End, and one-click preset buttons.
 */

export function initImageComparator({ stageEl, overlayEl, dividerEl, handleEl, readoutsEl, presetButtons }) {
  let pct = 50;
  let dragging = false;

  function setPosition(nextPct) {
    pct = Math.min(100, Math.max(0, Math.round(nextPct)));
    overlayEl.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    dividerEl.style.left = `${pct}%`;
    handleEl.style.left = `${pct}%`;
    handleEl.setAttribute("aria-valuenow", String(pct));
    if (readoutsEl) {
      readoutsEl.textContent = `Clean ${pct}% | Degraded ${100 - pct}%`;
    }
    presetButtons.forEach((btn) => {
      const target = Number(btn.dataset.splitPct);
      btn.setAttribute("aria-pressed", String(target === pct));
    });
  }

  function updateFromClientX(clientX) {
    const rect = stageEl.getBoundingClientRect();
    if (rect.width <= 0) return;
    const ratio = ((clientX - rect.left) / rect.width) * 100;
    setPosition(ratio);
  }

  stageEl.addEventListener("pointerdown", (e) => {
    dragging = true;
    stageEl.setPointerCapture?.(e.pointerId);
    updateFromClientX(e.clientX);
  });

  stageEl.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    updateFromClientX(e.clientX);
  });

  stageEl.addEventListener("pointerup", () => {
    dragging = false;
  });

  stageEl.addEventListener("pointercancel", () => {
    dragging = false;
  });

  handleEl.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 10 : 5;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      setPosition(pct - step);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      setPosition(pct + step);
    } else if (e.key === "Home") {
      e.preventDefault();
      setPosition(0);
    } else if (e.key === "End") {
      e.preventDefault();
      setPosition(100);
    }
  });

  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      setPosition(Number(btn.dataset.splitPct));
    });
  });

  setPosition(50);
  return { setPosition };
}
