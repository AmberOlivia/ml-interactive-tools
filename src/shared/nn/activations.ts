export type ActivationName = 'tanh' | 'relu' | 'sigmoid' | 'linear';

export interface Activation {
  name: ActivationName;
  fn: (x: number) => number;
}

export const ACTIVATIONS: Record<ActivationName, Activation> = {
  tanh: { name: 'tanh', fn: Math.tanh },
  relu: { name: 'relu', fn: (x) => Math.max(0, x) },
  sigmoid: { name: 'sigmoid', fn: (x) => 1 / (1 + Math.exp(-x)) },
  linear: { name: 'linear', fn: (x) => x },
};

export const ACTIVATION_NAMES: ActivationName[] = ['tanh', 'relu', 'sigmoid', 'linear'];
