import { useMemo } from 'react';
import { Network } from '../nn/network';
import { divergingColor, weightColor, weightWidth } from './colors';

export type GraphSelection =
  | { kind: 'edge'; layer: number; neuron: number; weightIndex: number }
  | { kind: 'bias'; layer: number; neuron: number }
  | null;

interface Props {
  network: Network;
  inputLabels: string[];
  // Activations for each layer (from forward()): index 0 = inputs, last = output.
  layerActivations: number[][] | null;
  // Which layers are "active" (already propagated). For step-forward animation.
  activeLayerIndex: number;
  onNeuronHover?: (layer: number, neuron: number) => void;
  onNeuronLeave?: () => void;
  hovered?: { layer: number; neuron: number } | null;
  // Click-to-edit
  selection?: GraphSelection;
  onEdgeClick?: (layer: number, neuron: number, weightIndex: number) => void;
  onNeuronClick?: (layer: number, neuron: number) => void;
  onBackgroundClick?: () => void;
}

const NEURON_R = 16;
const WIDTH = 760;
const HEIGHT = 340;
const PAD_X = 80;
const PAD_Y = 28;

function edgeMatches(
  s: GraphSelection,
  li: number,
  ni: number,
  wi: number,
): boolean {
  return (
    !!s &&
    s.kind === 'edge' &&
    s.layer === li &&
    s.neuron === ni &&
    s.weightIndex === wi
  );
}

function biasMatches(s: GraphSelection, li: number, ni: number): boolean {
  return !!s && s.kind === 'bias' && s.layer === li && s.neuron === ni;
}

export function NetworkGraph({
  network,
  inputLabels,
  layerActivations,
  activeLayerIndex,
  onNeuronHover,
  onNeuronLeave,
  hovered,
  selection,
  onEdgeClick,
  onNeuronClick,
  onBackgroundClick,
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
      onClick={(e) => {
        // Only fire background-click when the svg itself was clicked,
        // not bubbled from a child group.
        if (e.target === e.currentTarget) onBackgroundClick?.();
      }}
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
                const selected = edgeMatches(selection ?? null, li, ni, wi);
                return (
                  <g
                    key={`${li}-${ni}-${wi}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onEdgeClick?.(li, ni, wi);
                    }}
                    style={{ cursor: onEdgeClick ? 'pointer' : 'default' }}
                  >
                    {/* Invisible wider hit target */}
                    <line
                      x1={from.x + NEURON_R}
                      y1={from.y}
                      x2={to.x - NEURON_R}
                      y2={to.y}
                      stroke="transparent"
                      strokeWidth={10}
                    />
                    <line
                      x1={from.x + NEURON_R}
                      y1={from.y}
                      x2={to.x - NEURON_R}
                      y2={to.y}
                      stroke={
                        selected
                          ? '#0f172a'
                          : edgesActive
                            ? weightColor(w)
                            : '#d4d4d8'
                      }
                      strokeWidth={
                        selected
                          ? Math.max(3, weightWidth(w, maxAbsWeight) + 1.5)
                          : weightWidth(w, maxAbsWeight)
                      }
                      opacity={edgesActive || selected ? 0.95 : 0.25}
                      strokeDasharray={selected ? '6 3' : undefined}
                    />
                  </g>
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
            active && layerActivations ? (layerActivations[li]?.[ni] ?? 0) : 0;
          const fill = active ? divergingColor(activation) : '#ffffff';
          const isHovered =
            hovered && hovered.layer === li && hovered.neuron === ni;
          const biasSelected = biasMatches(selection ?? null, li, ni);
          const isInput = li === 0;
          return (
            <g
              key={`n-${li}-${ni}`}
              onMouseEnter={() => onNeuronHover?.(li, ni)}
              onMouseLeave={() => onNeuronLeave?.()}
              onClick={(e) => {
                e.stopPropagation();
                if (!isInput) onNeuronClick?.(li, ni);
              }}
              style={{ cursor: onNeuronClick && !isInput ? 'pointer' : 'default' }}
            >
              <circle
                cx={pos.x}
                cy={pos.y}
                r={NEURON_R}
                fill={fill}
                stroke={
                  biasSelected
                    ? '#0f172a'
                    : isHovered
                      ? '#0f172a'
                      : active
                        ? '#525252'
                        : '#a1a1aa'
                }
                strokeWidth={biasSelected ? 3 : isHovered ? 3 : 1.5}
                strokeDasharray={biasSelected ? '4 2' : undefined}
              />
              {li === 0 && inputLabels[ni] && (
                <text
                  x={pos.x - NEURON_R - 8}
                  y={pos.y + 4}
                  textAnchor="end"
                  fontSize={12}
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
                  style={{ pointerEvents: 'none' }}
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
