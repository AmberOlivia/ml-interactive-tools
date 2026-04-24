import { ACTIVATIONS, ActivationName } from './activations';
import { createRng, gaussian } from './random';

export interface Neuron {
  bias: number;
  weights: number[]; // length = previous layer size
  z: number; // pre-activation
  a: number; // post-activation
  // Captured at build time so the UI can offer "reset to initial"
  initialBias: number;
  initialWeights: number[];
}

export interface Layer {
  neurons: Neuron[];
  activation: ActivationName;
}

export interface Network {
  inputSize: number;
  layers: Layer[]; // hidden layers + output layer
  outputActivation: ActivationName;
}

export interface BuildOptions {
  inputSize: number;
  hiddenSizes: number[];
  hiddenActivation: ActivationName;
  outputActivation: ActivationName;
  seed: number;
}

export function buildNetwork(opts: BuildOptions): Network {
  const rng = createRng(opts.seed);
  const layerSizes = [opts.inputSize, ...opts.hiddenSizes, 1];
  const layers: Layer[] = [];

  for (let li = 1; li < layerSizes.length; li++) {
    const prevSize = layerSizes[li - 1];
    const size = layerSizes[li];
    const isOutput = li === layerSizes.length - 1;
    // He-ish scale: works reasonably for tanh/relu in a toy setting
    const scale = Math.sqrt(2 / Math.max(prevSize, 1));
    const neurons: Neuron[] = [];
    for (let n = 0; n < size; n++) {
      const weights: number[] = [];
      for (let w = 0; w < prevSize; w++) {
        weights.push(gaussian(rng) * scale);
      }
      neurons.push({
        bias: 0,
        weights,
        z: 0,
        a: 0,
        initialBias: 0,
        initialWeights: weights.slice(),
      });
    }
    layers.push({
      neurons,
      activation: isOutput ? opts.outputActivation : opts.hiddenActivation,
    });
  }

  return {
    inputSize: opts.inputSize,
    layers,
    outputActivation: opts.outputActivation,
  };
}

// Forward-pass result: activation value per layer (including the input layer at index 0).
export type LayerActivations = number[][];

export function forward(net: Network, inputs: number[]): LayerActivations {
  const result: LayerActivations = [inputs.slice()];
  let prev = inputs;
  for (const layer of net.layers) {
    const activationFn = ACTIVATIONS[layer.activation].fn;
    const out: number[] = [];
    for (const neuron of layer.neurons) {
      let z = neuron.bias;
      for (let i = 0; i < prev.length; i++) {
        z += neuron.weights[i] * prev[i];
      }
      const a = activationFn(z);
      neuron.z = z;
      neuron.a = a;
      out.push(a);
    }
    result.push(out);
    prev = out;
  }
  return result;
}

export function predict(net: Network, inputs: number[]): number {
  const acts = forward(net, inputs);
  return acts[acts.length - 1][0];
}

// Convenience for the graph viz
export interface NetworkShape {
  layerSizes: number[]; // [inputSize, hidden1, ..., outputSize]
}

export function getShape(net: Network): NetworkShape {
  return {
    layerSizes: [net.inputSize, ...net.layers.map((l) => l.neurons.length)],
  };
}

// Per-neuron activation over a 2D grid of inputs — used to render
// "mini-heatmaps" inside each neuron (Playground-style).
// Shape: grids[layer][neuron][gy][gx]
// Layer 0 = inputs; last = output.
export type NeuronGrid = number[][];
export type NetworkGrids = NeuronGrid[][];

// Compute each neuron's activation over a 2D grid of input points.
// `inputBuilder(x1, x2)` returns the input vector for domain point (x1, x2) —
// the caller decides how features and constants map to inputs.
export function computeNeuronGrids(
  net: Network,
  inputBuilder: (x1: number, x2: number) => number[],
  domain: { min: number; max: number },
  resolution: number,
): NetworkGrids {
  const layerSizes = [net.inputSize, ...net.layers.map((l) => l.neurons.length)];
  const grids: NetworkGrids = layerSizes.map((size) =>
    Array.from({ length: size }, () =>
      Array.from({ length: resolution }, () => new Array(resolution).fill(0)),
    ),
  );
  const span = domain.max - domain.min;
  for (let gy = 0; gy < resolution; gy++) {
    const x2 = domain.max - ((gy + 0.5) / resolution) * span;
    for (let gx = 0; gx < resolution; gx++) {
      const x1 = domain.min + ((gx + 0.5) / resolution) * span;
      const inputs = inputBuilder(x1, x2);
      const acts = forward(net, inputs);
      // Guard against size mismatch: can happen for one render after feature
      // count changes but before the network rebuild effect has fired.
      const layerLimit = Math.min(acts.length, grids.length);
      for (let li = 0; li < layerLimit; li++) {
        const neuronLimit = Math.min(acts[li].length, grids[li].length);
        for (let ni = 0; ni < neuronLimit; ni++) {
          grids[li][ni][gy][gx] = acts[li][ni];
        }
      }
    }
  }
  return grids;
}
