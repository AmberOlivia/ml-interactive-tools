import { ACTIVATIONS, ActivationName } from './activations';
import { createRng, gaussian } from './random';

export interface Neuron {
  bias: number;
  weights: number[]; // length = previous layer size
  z: number; // pre-activation
  a: number; // post-activation
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
