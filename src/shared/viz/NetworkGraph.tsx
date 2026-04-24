import { useMemo } from 'react';
import { Network } from '../nn/network';
import { divergingColor, weightColor, weightWidth } from './colors';

interface Props {
  network: Network;
  inputLabels: string[];
  // Activations for each layer (from forward()): index 0 = inputs, last = output.
  layerActivations: number[][] | null;
  // Which layers are "active" (already propagated). For step-forward animation.
  // e.g. activeLayerIndex=0 means only the input layer is active.
  activeLayerIndex: number;
  onNeuronHover?: (layer: number, neuron: number) => void;
  onNeuronLeave?: () => void;
  hovered?: { layer: number; neuron: number } | null;
}

const NEURON_R = 16;
const WIDTH = 760;
const HEIGHT = 340;
const PAD_X = 80;
const PAD_Y = 28;

export function NetworkGraph({
  network,
  inputLabels,
  layerActivations,
  activeLayerIndex,
  onNeuronHover,
  onNeuronLeave,
  hovered,
}: Props) {
  const layerSizes = useMemo(
    () => [network.inputSize, ...network.layers.map((l) => l.neurons.length)],
    [network],
  );

  const positions = useMemo(() => {
    const cols = layerSizes.length;
    const colX = (i: number) =>
      cols === 1 ? WIDTH / 2 : PAD_X + (i * (WIDTH - 2 * PAD_X)) / (cols - 1);
    return layerSizes.map((size, li) => {
      const x = colX(li);
      const h = HEIGHT - 2 * PAD_Y;
      return Array.from({ length: size }, (_, ni) => {
        const y = size === 1 ? HEIGHT / 2 : PAD_Y + (ni * h) / (size - 1);
        return { x, y };
      });
    });
  }, [layerSizes]);

  const maxAbsWeight = useMemo(() => {
    let m = 0;
    for (const layer of network.layers) {
      for (const n of layer.neurons) {
        for (const w of n.weights) m = Math.max(m, Math.abs(w));
      }
    }
    return m;
  }, [network]);

  return (
    <svg
      width={WIDTH}
      height={HEIGHT}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      style={{ background: '#fafafa', borderRadius: 8, border: '1px solid #e5e7eb' }}
    >
      {/* Edges: layer l-1 → l (l from 1..layers.length) */}
      {network.layers.map((layer, li) => {
        const fromLayerIdx = li; // positions index of previous layer (inputs are index 0)
        const toLayerIdx = li + 1;
        const edgesActive = activeLayerIndex >= toLayerIdx;
        return (
          <g key={`edges-${li}`}>
            {layer.neurons.map((neuron, ni) =>
              neuron.weights.map((w, wi) => {
                const from = positions[fromLayerIdx][wi];
                const to = positions[toLayerIdx][ni];
                return (
                  <line
                    key={`${li}-${ni}-${wi}`}
                    x1={from.x + NEURON_R}
                    y1={from.y}
                    x2={to.x - NEURON_R}
                    y2={to.y}
                    stroke={edgesActive ? weightColor(w) : '#d4d4d8'}
                    strokeWidth={weightWidth(w, maxAbsWeight)}
                    opacity={edgesActive ? 0.9 : 0.25}
                  />
                );
              }),
            )}
          </g>
        );
      })}

      {/* Neurons */}
      {positions.map((layer, li) =>
        layer.map((pos, ni) => {
          const active = activeLayerIndex >= li;
          const activation =
            active && layerActivations ? layerActivations[li]?.[ni] ?? 0 : 0;
          const fill = active ? divergingColor(activation) : '#ffffff';
          const isHovered =
            hovered && hovered.layer === li && hovered.neuron === ni;
          return (
            <g
              key={`n-${li}-${ni}`}
              onMouseEnter={() => onNeuronHover?.(li, ni)}
              onMouseLeave={() => onNeuronLeave?.()}
              style={{ cursor: 'pointer' }}
            >
              <circle
                cx={pos.x}
                cy={pos.y}
                r={NEURON_R}
                fill={fill}
                stroke={isHovered ? '#0f172a' : active ? '#525252' : '#a1a1aa'}
                strokeWidth={isHovered ? 3 : 1.5}
              />
              {li === 0 && inputLabels[ni] && (
                <text
                  x={pos.x - NEURON_R - 8}
                  y={pos.y + 4}
                  textAnchor="end"
                  fontSize={13}
                  fill="#334155"
                >
                  {inputLabels[ni]}
                </text>
              )}
              {active && layerActivations && (
                <text
                  x={pos.x}
                  y={pos.y + 4}
                  textAnchor="middle"
                  fontSize={10}
                  fill={Math.abs(activation) > 0.6 ? '#fff' : '#1f2937'}
                >
                  {activation.toFixed(2)}
                </text>
              )}
            </g>
          );
        }),
      )}

      {/* Layer labels */}
      {positions.map((layer, li) => {
        const label =
          li === 0
            ? 'Input'
            : li === positions.length - 1
              ? 'Output'
              : `Hidden ${li}`;
        return (
          <text
            key={`label-${li}`}
            x={layer[0].x}
            y={18}
            textAnchor="middle"
            fontSize={12}
            fill="#64748b"
          >
            {label}
          </text>
        );
      })}
    </svg>
  );
}

export { WIDTH as GRAPH_WIDTH, HEIGHT as GRAPH_HEIGHT };
