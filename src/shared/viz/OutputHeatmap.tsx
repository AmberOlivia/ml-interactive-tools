import { useEffect, useRef } from 'react';
import { Network, forward } from '../nn/network';
import { computeFeatures, ConstantValues, FeatureName } from '../nn/features';
import { DOMAIN, Sample, ProblemType } from '../datasets';

interface Props {
  network: Network;
  features: FeatureName[];
  constants: ConstantValues;
  samples: Sample[];
  problem: ProblemType;
  // Which neuron's activation to render. layer = positions index (0 = input,
  // last = output). null/undefined renders the network's final output.
  target?: { layer: number; neuron: number } | null;
  size?: number;
  gridResolution?: number;
}

const DEFAULT_SIZE = 360;
const DEFAULT_RES = 60; // 60x60 grid cells

// Diverging color matching NetworkGraph
function color(v: number): [number, number, number] {
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    return [
      Math.round(255 + t * (245 - 255)),
      Math.round(255 + t * (158 - 255)),
      Math.round(255 + t * (11 - 255)),
    ];
  }
  const s = -t;
  return [
    Math.round(255 + s * (59 - 255)),
    Math.round(255 + s * (130 - 255)),
    Math.round(255 + s * (246 - 255)),
  ];
}

export function OutputHeatmap({
  network,
  features,
  constants,
  samples,
  problem,
  target,
  size = DEFAULT_SIZE,
  gridResolution = DEFAULT_RES,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Render background heatmap at low resolution, scale up.
    const cellPx = size / gridResolution;
    const domainSpan = DOMAIN.max - DOMAIN.min;

    for (let gy = 0; gy < gridResolution; gy++) {
      for (let gx = 0; gx < gridResolution; gx++) {
        // Map grid cell center to domain coords. Y increases downward on canvas.
        const x1 =
          DOMAIN.min + ((gx + 0.5) / gridResolution) * domainSpan;
        const x2 =
          DOMAIN.max - ((gy + 0.5) / gridResolution) * domainSpan;
        const inputs = computeFeatures(x1, x2, features, constants);
        const acts = forward(network, inputs);
        const out = target
          ? (acts[target.layer]?.[target.neuron] ?? 0)
          : acts[acts.length - 1][0];
        const [r, g, b] = color(out);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(gx * cellPx, gy * cellPx, cellPx + 1, cellPx + 1);
      }
    }

    // Axes
    ctx.strokeStyle = 'rgba(0,0,0,0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, size / 2);
    ctx.lineTo(size, size / 2);
    ctx.moveTo(size / 2, 0);
    ctx.lineTo(size / 2, size);
    ctx.stroke();

    // Data points
    for (const s of samples) {
      const px = ((s.x1 - DOMAIN.min) / domainSpan) * size;
      const py = ((DOMAIN.max - s.x2) / domainSpan) * size;
      const [r, g, b] = color(s.label);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.strokeStyle = 'rgba(0,0,0,0.65)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(px, py, problem === 'classification' ? 4 : 3, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }, [network, features, constants, samples, problem, target, size, gridResolution]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      style={{
        borderRadius: 8,
        border: '1px solid #e5e7eb',
        imageRendering: 'pixelated',
      }}
    />
  );
}
