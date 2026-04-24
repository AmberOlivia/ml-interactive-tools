import { useMemo } from 'react';
import { Network, NetworkGrids } from '../nn/network';
import { divergingColor, weightColor, weightWidth } from './colors';

// Indexing convention:
// - 'edge'   layer = network.layers index (0..N-1). Edge feeds positions[layer+1][neuron].
// - 'neuron' layer = positions index (0..N). 0 = input layer. Bias lives at
//            network.layers[layer-1].neurons[neuron] for layer > 0.
export type GraphSelection =
  | { kind: 'edge'; layer: number; neuron: number; weightIndex: number }
  | { kind: 'neuron'; layer: number; neuron: number }
  | null;

interface Props {
  network: Network;
  inputLabels: string[];
  layerActivations: number[][] | null;
  activeLayerIndex: number;
  // Per-neuron activation grids over the 2D input domain. Used for mini-heatmaps.
  grids: NetworkGrids | null;
  onNeuronHover?: (layer: number, neuron: number) => void;
  onNeuronLeave?: () => void;
  hovered?: { layer: number; neuron: number } | null;
  selection?: GraphSelection;
  onEdgeClick?: (layer: number, neuron: number, weightIndex: number) => void;
  onNeuronClick?: (layer: number, neuron: number) => void;
  onBackgroundClick?: () => void;
}

const NEURON_R = 16; // half-side of the square neuron tile
const NEURON_SIZE = NEURON_R * 2;
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

function neuronMatches(s: GraphSelection, li: number, ni: number): boolean {
  return !!s && s.kind === 'neuron' && s.layer === li && s.neuron === ni;
}

function gridToDataUrl(grid: number[][]): string {
  const res = grid.length;
  const canvas = document.createElement('canvas');
  canvas.width = res;
  canvas.height = res;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  for (let gy = 0; gy < res; gy++) {
    for (let gx = 0; gx < res; gx++) {
      ctx.fillStyle = divergingColor(grid[gy][gx]);
      ctx.fillRect(gx, gy, 1, 1);
    }
  }
  return canvas.toDataURL();
}

// One mini heatmap as an SVG <image>. Memoizes the data URL on grid identity.
function NeuronTile({
  x,
  y,
  size,
  grid,
  visible,
}: {
  x: number;
  y: number;
  size: number;
  grid: number[][] | null;
  visible: boolean;
}) {
  const dataUrl = useMemo(
    () => (grid ? gridToDataUrl(grid) : ''),
    [grid],
  );
  if (!visible || !grid) {
    return (
      <rect
        x={x}
        y={y}
        width={size}
        height={size}
        fill="#ffffff"
        pointerEvents="none"
      />
    );
  }
  return (
    <image
      href={dataUrl}
      x={x}
      y={y}
      width={size}
      height={size}
      preserveAspectRatio="none"
      style={{ imageRendering: 'pixelated' }}
      pointerEvents="none"
    />
  );
}

export function NetworkGraph({
  network,
  inputLabels,
  layerActivations,
  activeLayerIndex,
  grids,
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
        if (e.target === e.currentTarget) onBackgroundClick?.();
      }}
      style={{ background: '#fafafa', borderRadius: 8, border: '1px solid #e5e7eb' }}
    >
      {/* Edges */}
      {network.layers.map((layer, li) => {
        const fromLayerIdx = li;
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
          const isHovered =
            hovered && hovered.layer === li && hovered.neuron === ni;
          const neuronSelected = neuronMatches(selection ?? null, li, ni);
          const grid = grids?.[li]?.[ni] ?? null;
          return (
            <g
              key={`n-${li}-${ni}`}
              onMouseEnter={() => onNeuronHover?.(li, ni)}
              onMouseLeave={() => onNeuronLeave?.()}
              onClick={(e) => {
                e.stopPropagation();
                onNeuronClick?.(li, ni);
              }}
              style={{ cursor: onNeuronClick ? 'pointer' : 'default' }}
            >
              {/* clip the heatmap to the neuron tile */}
              <clipPath id={`tile-${li}-${ni}`}>
                <rect
                  x={pos.x - NEURON_R}
                  y={pos.y - NEURON_R}
                  width={NEURON_SIZE}
                  height={NEURON_SIZE}
                  rx={4}
                  ry={4}
                />
              </clipPath>
              <g clipPath={`url(#tile-${li}-${ni})`}>
                <NeuronTile
                  x={pos.x - NEURON_R}
                  y={pos.y - NEURON_R}
                  size={NEURON_SIZE}
                  grid={grid}
                  visible={active}
                />
              </g>
              <rect
                x={pos.x - NEURON_R}
                y={pos.y - NEURON_R}
                width={NEURON_SIZE}
                height={NEURON_SIZE}
                rx={4}
                ry={4}
                fill="transparent"
                stroke={
                  neuronSelected
                    ? '#0f172a'
                    : isHovered
                      ? '#0f172a'
                      : active
                        ? '#525252'
                        : '#a1a1aa'
                }
                strokeWidth={neuronSelected || isHovered ? 2 : 1.25}
                strokeDasharray={neuronSelected ? '4 2' : undefined}
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
                  fontWeight={600}
                  fill={Math.abs(activation) > 0.4 ? '#fff' : '#0f172a'}
                  stroke={Math.abs(activation) > 0.4 ? '#0f172a' : '#ffffff'}
                  strokeWidth={0.4}
                  paintOrder="stroke"
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
