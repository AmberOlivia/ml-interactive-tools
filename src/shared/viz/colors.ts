// Diverging color scale: negative → blue, zero → white, positive → orange.
// Matches the TF Playground aesthetic so students can map concepts across tools.
export function divergingColor(v: number): string {
  // Clamp to [-1, 1]
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    // white → orange (#f59e0b)
    const r = Math.round(255 + t * (245 - 255));
    const g = Math.round(255 + t * (158 - 255));
    const b = Math.round(255 + t * (11 - 255));
    return `rgb(${r},${g},${b})`;
  } else {
    // white → blue (#3b82f6)
    const s = -t;
    const r = Math.round(255 + s * (59 - 255));
    const g = Math.round(255 + s * (130 - 255));
    const b = Math.round(255 + s * (246 - 255));
    return `rgb(${r},${g},${b})`;
  }
}

export function weightColor(w: number): string {
  // Stronger contrast for edges — saturate even mid-magnitude weights.
  const clamped = Math.max(-1, Math.min(1, w));
  return divergingColor(clamped);
}

export function weightWidth(w: number, maxAbs: number): number {
  const normalized = maxAbs > 0 ? Math.abs(w) / maxAbs : 0;
  return 0.5 + normalized * 3.5;
}
