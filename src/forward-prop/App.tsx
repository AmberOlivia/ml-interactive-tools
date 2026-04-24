import { useMemo, useState, useEffect, useRef } from 'react';
import {
  buildNetwork,
  computeNeuronGrids,
  forward,
  Network,
} from '../shared/nn/network';
import { ACTIVATION_NAMES, ActivationName } from '../shared/nn/activations';
import {
  FEATURE_NAMES,
  FeatureName,
  FEATURES,
  computeFeatures,
  ConstantValues,
} from '../shared/nn/features';
import {
  generateDataset,
  DATASET_NAMES,
  DatasetName,
  ProblemType,
  DOMAIN,
} from '../shared/datasets';
import { NetworkGraph, GraphSelection } from '../shared/viz/NetworkGraph';
import { OutputHeatmap } from '../shared/viz/OutputHeatmap';
import { Slider } from '../shared/ui/Slider';
import { divergingColor } from '../shared/viz/colors';

export function App() {
  // Problem / dataset
  const [problem, setProblem] = useState<ProblemType>('classification');
  const [datasetName, setDatasetName] = useState<DatasetName>('circle');
  const [noise, setNoise] = useState(0.1);
  const [dataSeed, setDataSeed] = useState(1);
  const sampleCount = 200;

  // Features (input transformations)
  const [features, setFeatures] = useState<FeatureName[]>(['x1', 'x2']);
  // x0 is the bias baseline input (= 1, per the course's "x0 = 1" convention).
  // Not a user-adjustable value — its weights (θ₀) are what the student edits.
  const [constants] = useState<ConstantValues>({ x0: 1 });

  // Architecture
  const [hiddenSizes, setHiddenSizes] = useState<number[]>([4, 2]);
  const [hiddenActivation, setHiddenActivation] = useState<ActivationName>('tanh');
  const outputActivation: ActivationName =
    problem === 'classification' ? 'tanh' : 'linear';
  const [weightSeed, setWeightSeed] = useState(0.12);

  // Step-forward state: which layer's activations are revealed
  const [activeLayer, setActiveLayer] = useState(0);

  // Test input (click on heatmap to set)
  const [testInput, setTestInput] = useState<[number, number]>([1.5, 1.5]);

  // Hovered neuron (for inspector panel)
  const [hovered, setHovered] = useState<{ layer: number; neuron: number } | null>(
    null,
  );

  // Derived state
  const samples = useMemo(
    () =>
      generateDataset({
        name: datasetName,
        count: sampleCount,
        noise,
        seed: dataSeed,
        problem,
      }),
    [datasetName, noise, dataSeed, problem],
  );

  // Network lives in state so users can click-to-edit weights and biases.
  // It gets rebuilt when architecture or seed changes.
  const [network, setNetwork] = useState<Network>(() =>
    buildNetwork({
      inputSize: features.length,
      hiddenSizes,
      hiddenActivation,
      outputActivation,
      seed: weightSeed,
    }),
  );
  const [selection, setSelection] = useState<GraphSelection>(null);

  useEffect(() => {
    setNetwork(
      buildNetwork({
        inputSize: features.length,
        hiddenSizes,
        hiddenActivation,
        outputActivation,
        seed: weightSeed,
      }),
    );
    setSelection(null);
  }, [features.length, hiddenSizes, hiddenActivation, outputActivation, weightSeed]);

  const editWeight = (layer: number, neuron: number, weightIdx: number, value: number) => {
    setNetwork((prev) => {
      const next = structuredClone(prev);
      next.layers[layer].neurons[neuron].weights[weightIdx] = value;
      return next;
    });
  };

  const resetSelectedWeight = () => {
    if (!selection || selection.kind !== 'edge') return;
    const n = network.layers[selection.layer].neurons[selection.neuron];
    editWeight(selection.layer, selection.neuron, selection.weightIndex, n.initialWeights[selection.weightIndex]);
  };

  const testInputs = useMemo(
    () => computeFeatures(testInput[0], testInput[1], features, constants),
    [testInput, features, constants],
  );

  // Per-neuron activation grids — used for mini heatmaps inside each neuron.
  // Computed first (it mutates z/a during the grid sweep), then forward()
  // runs again on the test input to leave the network in the probe state.
  const grids = useMemo(
    () =>
      computeNeuronGrids(
        network,
        (x1, x2) => computeFeatures(x1, x2, features, constants),
        DOMAIN,
        20,
      ),
    [network, features, constants],
  );

  const layerActivations = useMemo(
    () => forward(network, testInputs),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [network, testInputs, grids],
  );

  const maxLayerIndex = layerActivations.length - 1;

  // Reset step animation when architecture changes
  useEffect(() => {
    setActiveLayer(0);
  }, [network]);

  // Auto-play animation
  const playingRef = useRef<number | null>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    playingRef.current = window.setInterval(() => {
      setActiveLayer((l) => {
        if (l >= maxLayerIndex) {
          setPlaying(false);
          return l;
        }
        return l + 1;
      });
    }, 650);
    return () => {
      if (playingRef.current) window.clearInterval(playingRef.current);
    };
  }, [playing, maxLayerIndex]);

  const handlePlay = () => {
    if (activeLayer >= maxLayerIndex) setActiveLayer(0);
    setPlaying(true);
  };
  const handleStep = () => {
    setPlaying(false);
    setActiveLayer((l) => Math.min(l + 1, maxLayerIndex));
  };
  const handleStepBack = () => {
    setPlaying(false);
    setActiveLayer((l) => Math.max(l - 1, 0));
  };
  const handleReset = () => {
    setPlaying(false);
    setActiveLayer(0);
  };
  const handleFull = () => {
    setPlaying(false);
    setActiveLayer(maxLayerIndex);
  };

  const toggleFeature = (name: FeatureName) => {
    setFeatures((curr) =>
      curr.includes(name)
        ? curr.filter((f) => f !== name).length > 0
          ? curr.filter((f) => f !== name)
          : curr
        : [...curr, name],
    );
  };

  const setHiddenLayerCount = (n: number) => {
    setHiddenSizes((curr) => {
      const next = curr.slice(0, n);
      while (next.length < n) next.push(3);
      return next;
    });
  };

  const setLayerSize = (i: number, size: number) => {
    setHiddenSizes((curr) => curr.map((s, idx) => (idx === i ? size : s)));
  };

  const output = layerActivations[layerActivations.length - 1][0];

  // What neuron's heatmap to show in the big OutputHeatmap.
  // Selection 'neuron' uses positions index directly. 'edge' promotes the
  // destination neuron (layer + 1 in positions space) so students see what
  // that edge contributes to.
  const viewedTarget: { layer: number; neuron: number } | null = !selection
    ? null
    : selection.kind === 'neuron'
      ? { layer: selection.layer, neuron: selection.neuron }
      : { layer: selection.layer + 1, neuron: selection.neuron };

  const viewedValueAtProbe = viewedTarget
    ? (layerActivations[viewedTarget.layer]?.[viewedTarget.neuron] ?? 0)
    : output;

  const captionForTarget = (() => {
    if (!viewedTarget) return 'Network output over input space — click to place probe';
    if (viewedTarget.layer === 0) {
      return `Input feature: ${FEATURES[features[viewedTarget.neuron]].label}`;
    }
    const last = layerActivations.length - 1;
    if (viewedTarget.layer === last) return 'Output neuron';
    return `Hidden layer ${viewedTarget.layer}, neuron ${viewedTarget.neuron + 1}`;
  })();

  // Heatmap click handler — map pixel to domain coords
  const heatmapSize = 360;
  const handleHeatmapClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const span = DOMAIN.max - DOMAIN.min;
    const x1 = DOMAIN.min + (px / heatmapSize) * span;
    const x2 = DOMAIN.max - (py / heatmapSize) * span;
    setTestInput([x1, x2]);
  };

  const inputLabels = features.map(
    (f, i) => `${FEATURES[f].label} = ${layerActivations[0][i].toFixed(2)}`,
  );

  // Build neuron-inspector data
  const inspector = (() => {
    if (!hovered) return null;
    const { layer, neuron } = hovered;
    if (layer === 0) {
      return {
        title: `Input: ${inputLabels[neuron]}`,
        rows: [['value', layerActivations[0][neuron].toFixed(3)]],
      };
    }
    const n = network.layers[layer - 1]?.neurons[neuron];
    if (!n) return null;
    const prevAct = layerActivations[layer - 1];
    const active = activeLayer >= layer;
    const rows: [string, string][] = [
      ['bias (b)', n.bias.toFixed(3)],
      ['pre-act (z)', active ? n.z.toFixed(3) : '—'],
      ['post-act (a)', active ? n.a.toFixed(3) : '—'],
      ['activation', network.layers[layer - 1].activation],
    ];
    const weightRows: [string, string][] = n.weights.map((w, wi) => {
      const prev = prevAct[wi];
      return [
        `w${wi + 1} · a_prev${wi + 1}`,
        `${w.toFixed(2)} · ${active ? prev.toFixed(2) : '—'}`,
      ];
    });
    return {
      title: `${layer === maxLayerIndex ? 'Output neuron' : `Hidden L${layer} · N${neuron + 1}`}`,
      rows: [...rows, ['—', '—'], ...weightRows],
    };
  })();

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Forward Propagation</h1>
          <div className="subtitle">
            Machine Learning for Engineers — interactive tool
          </div>
        </div>
        <nav>
          <a href="./index.html">← All tools</a>
          <a href="./backprop.html">Backprop →</a>
        </nav>
      </header>

      <p className="intro">
        Tune the inputs, architecture, and random seed — then click{' '}
        <b>Step →</b> to propagate signals layer by layer. Click anywhere on the
        heatmap to pick a test point (x₁, x₂); the network graph shows the
        activation values that point produces. Hover a neuron to inspect its
        weights and computation.
      </p>

      <div className="layout">
        {/* LEFT: data + features */}
        <div>
          <div className="panel">
            <h3>Problem</h3>
            <div className="control-row">
              <label>Type</label>
              <select
                value={problem}
                onChange={(e) => setProblem(e.target.value as ProblemType)}
              >
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>
          </div>
          <div className="panel" style={{ marginTop: 16 }}>
            <h3>Dataset</h3>
            <div className="control-row">
              <label>Pattern</label>
              <select
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value as DatasetName)}
              >
                {DATASET_NAMES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>
            <Slider
              label="Noise"
              value={noise}
              min={0}
              max={0.5}
              step={0.02}
              format={(v) => v.toFixed(2)}
              onChange={setNoise}
            />
            <Slider
              label="Data seed"
              value={dataSeed}
              min={1}
              max={20}
              onChange={setDataSeed}
            />
          </div>
          <div className="panel" style={{ marginTop: 16 }}>
            <h3>Input features</h3>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
              x₀ = 1 is the <b>bias baseline</b> (as in your course
              notes). Its weights into the next layer act as θ₀. x₁, x₂
              come from the data; the rest are engineered features built
              from x₁ and x₂.
            </div>
            <div className="feature-grid">
              {FEATURE_NAMES.map((f) => (
                <label key={f}>
                  <input
                    type="checkbox"
                    checked={features.includes(f)}
                    onChange={() => toggleFeature(f)}
                  />
                  {FEATURES[f].label}
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* CENTER: viz + controls */}
        <div className="center-col">
          <div className="button-row">
            <button onClick={handleReset} disabled={activeLayer === 0}>
              ⟲ Reset
            </button>
            <button onClick={handleStepBack} disabled={activeLayer === 0}>
              ← Step back
            </button>
            <button
              className="primary"
              onClick={handleStep}
              disabled={activeLayer >= maxLayerIndex}
            >
              Step → (layer {activeLayer} → {activeLayer + 1})
            </button>
            <button onClick={handlePlay} disabled={playing}>
              ▶ Play
            </button>
            <button onClick={handleFull} disabled={activeLayer >= maxLayerIndex}>
              ⇥ Full pass
            </button>
            <span className="step-indicator">
              {activeLayer === maxLayerIndex
                ? `Complete · output = ${output.toFixed(3)}`
                : `Showing layer ${activeLayer} of ${maxLayerIndex}`}
            </span>
          </div>

          <NetworkGraph
            network={network}
            inputLabels={inputLabels}
            layerActivations={layerActivations}
            activeLayerIndex={activeLayer}
            grids={grids}
            onNeuronHover={(l, n) => setHovered({ layer: l, neuron: n })}
            onNeuronLeave={() => setHovered(null)}
            hovered={hovered}
            selection={selection}
            onEdgeClick={(layer, neuron, weightIndex) =>
              setSelection({ kind: 'edge', layer, neuron, weightIndex })
            }
            onNeuronClick={(layer, neuron) =>
              setSelection({ kind: 'neuron', layer, neuron })
            }
            onBackgroundClick={() => setSelection(null)}
          />

          {selection && selection.kind === 'edge' && (
            <WeightEditor
              network={network}
              selection={selection}
              onWeightChange={editWeight}
              onReset={resetSelectedWeight}
              onClose={() => setSelection(null)}
            />
          )}

          <div className="output-row">
            <div>
              <div
                style={{
                  textAlign: 'center',
                  fontSize: 12,
                  color: viewedTarget ? '#0f172a' : '#64748b',
                  marginBottom: 6,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 10,
                }}
              >
                <span style={{ fontWeight: viewedTarget ? 600 : 400 }}>
                  Showing: {captionForTarget}
                </span>
                {viewedTarget && (
                  <button
                    onClick={() => setSelection(null)}
                    style={{ padding: '2px 8px', fontSize: 11 }}
                  >
                    ← back to output
                  </button>
                )}
              </div>
              <div
                style={{ cursor: 'crosshair', display: 'inline-block' }}
                onClick={handleHeatmapClick}
              >
                <OutputHeatmap
                  network={network}
                  features={features}
                  constants={constants}
                  samples={samples}
                  problem={problem}
                  target={viewedTarget}
                />
              </div>
              <TestInputMarker
                testInput={testInput}
                size={heatmapSize}
                value={viewedValueAtProbe}
              />
            </div>
          </div>

          <div className="legend">
            <span>
              <span
                className="swatch"
                style={{ background: divergingColor(-1) }}
              />{' '}
              −1
            </span>
            <span>
              <span
                className="swatch"
                style={{ background: divergingColor(0) }}
              />{' '}
              0
            </span>
            <span>
              <span
                className="swatch"
                style={{ background: divergingColor(1) }}
              />{' '}
              +1
            </span>
            <span style={{ marginLeft: 16 }}>
              Edge thickness = |weight|, color = sign
            </span>
          </div>
        </div>

        {/* RIGHT: architecture + inspector */}
        <div>
          <div className="panel">
            <h3>Architecture</h3>
            <Slider
              label="Hidden layers"
              value={hiddenSizes.length}
              min={0}
              max={4}
              onChange={setHiddenLayerCount}
            />
            {hiddenSizes.map((s, i) => (
              <Slider
                key={i}
                label={`Layer ${i + 1} neurons`}
                value={s}
                min={1}
                max={8}
                onChange={(v) => setLayerSize(i, v)}
              />
            ))}
            <div className="control-row">
              <label>Activation (hidden)</label>
              <select
                value={hiddenActivation}
                onChange={(e) =>
                  setHiddenActivation(e.target.value as ActivationName)
                }
              >
                {ACTIVATION_NAMES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>
            <Slider
              label="Weight seed"
              value={weightSeed}
              min={0}
              max={1}
              step={0.01}
              format={(v) => v.toFixed(2)}
              onChange={setWeightSeed}
            />
            <button
              style={{ width: '100%' }}
              onClick={() => setWeightSeed(Math.random())}
            >
              🎲 Reshuffle weights
            </button>
          </div>

          <div className="panel" style={{ marginTop: 16 }}>
            <h3>Neuron inspector</h3>
            {inspector ? (
              <div className="neuron-inspector">
                <div style={{ fontWeight: 600, marginBottom: 6 }}>
                  {inspector.title}
                </div>
                <table>
                  <tbody>
                    {inspector.rows.map(([k, v], i) => (
                      <tr key={i}>
                        <td>{k}</td>
                        <td>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', fontSize: 13 }}>
                Hover a neuron in the graph to see its weights and computation.
              </div>
            )}
          </div>

          <div className="panel" style={{ marginTop: 16 }}>
            <h3>Test point</h3>
            <div style={{ fontSize: 13, color: '#334155' }}>
              <div>x₁ = {testInput[0].toFixed(2)}</div>
              <div>x₂ = {testInput[1].toFixed(2)}</div>
              <div style={{ marginTop: 8, fontWeight: 600 }}>
                Output ŷ = {output.toFixed(3)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function WeightEditor({
  network,
  selection,
  onWeightChange,
  onReset,
  onClose,
}: {
  network: Network;
  selection: Extract<NonNullable<GraphSelection>, { kind: 'edge' }>;
  onWeightChange: (layer: number, neuron: number, weightIdx: number, v: number) => void;
  onReset: () => void;
  onClose: () => void;
}) {
  // selection.layer is the network.layers index (0..N-1).
  const networkLayerIdx = selection.layer;
  const totalLayers = network.layers.length;
  const neuron = network.layers[networkLayerIdx].neurons[selection.neuron];

  const currentValue = neuron.weights[selection.weightIndex];
  const initialValue = neuron.initialWeights[selection.weightIndex];

  const destLabel =
    networkLayerIdx === totalLayers - 1
      ? 'output'
      : `hidden ${networkLayerIdx + 1}, neuron ${selection.neuron + 1}`;
  const sourceLabel =
    networkLayerIdx === 0
      ? `input ${selection.weightIndex + 1}`
      : `hidden ${networkLayerIdx}, neuron ${selection.weightIndex + 1}`;

  const title = `Weight: ${sourceLabel} → ${destLabel}`;

  return (
    <div
      style={{
        width: 760,
        padding: '10px 14px',
        border: '1px solid #0f172a',
        borderRadius: 8,
        background: '#f1f5f9',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        fontSize: 13,
      }}
    >
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 600 }}>{title}</div>
        <div style={{ color: '#64748b', fontSize: 11 }}>
          initial: {initialValue.toFixed(3)}
        </div>
      </div>
      <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        value
        <input
          type="number"
          step={0.05}
          value={Number(currentValue.toFixed(4))}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isNaN(v)) return;
            onWeightChange(networkLayerIdx, selection.neuron, selection.weightIndex, v);
          }}
          style={{
            width: 90,
            padding: '4px 6px',
            border: '1px solid #cbd5e1',
            borderRadius: 4,
            fontFamily: 'ui-monospace, monospace',
          }}
        />
      </label>
      <input
        type="range"
        min={-3}
        max={3}
        step={0.01}
        value={currentValue}
        onChange={(e) => {
          const v = Number(e.target.value);
          onWeightChange(networkLayerIdx, selection.neuron, selection.weightIndex, v);
        }}
        style={{ width: 180 }}
      />
      <button onClick={onReset}>⟲ reset</button>
      <button onClick={onClose}>× close</button>
    </div>
  );
}

function TestInputMarker({
  testInput,
  size,
  value,
}: {
  testInput: [number, number];
  size: number;
  value: number;
}) {
  const span = DOMAIN.max - DOMAIN.min;
  const px = ((testInput[0] - DOMAIN.min) / span) * size;
  const py = ((DOMAIN.max - testInput[1]) / span) * size;
  return (
    <div
      style={{
        position: 'relative',
        pointerEvents: 'none',
        marginTop: -size,
        height: size,
        width: size,
      }}
    >
      <div
        style={{
          position: 'absolute',
          left: px - 8,
          top: py - 8,
          width: 16,
          height: 16,
          borderRadius: '50%',
          background: divergingColor(value),
          border: '2px solid #0f172a',
          boxShadow: '0 0 0 2px rgba(255,255,255,0.85)',
        }}
      />
    </div>
  );
}
