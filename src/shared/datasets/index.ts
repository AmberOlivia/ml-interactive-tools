import { createRng } from '../nn/random';

export type DatasetName = 'circle' | 'xor' | 'gaussian' | 'spiral';
export type ProblemType = 'classification' | 'regression';

export interface Sample {
  x1: number;
  x2: number;
  label: number; // classification: -1 or +1; regression: continuous in [-1, 1]
}

export const DATASET_NAMES: DatasetName[] = ['circle', 'xor', 'gaussian', 'spiral'];

export interface DatasetOptions {
  name: DatasetName;
  count: number;
  noise: number; // 0..0.5
  seed: number;
  problem: ProblemType;
}

// Domain: both axes in [-6, 6]
export const DOMAIN = { min: -6, max: 6 };

export function generateDataset(opts: DatasetOptions): Sample[] {
  const rng = createRng(opts.seed);
  const rand = () => rng() * 2 - 1; // [-1, 1]
  const samples: Sample[] = [];
  const { name, count, noise } = opts;

  for (let i = 0; i < count; i++) {
    let x1 = 0;
    let x2 = 0;
    let label = 0;

    if (name === 'circle') {
      const inner = i < count / 2;
      const r = inner ? rng() * 2 : 3 + rng() * 2;
      const theta = rng() * 2 * Math.PI;
      x1 = r * Math.cos(theta) + rand() * noise * 3;
      x2 = r * Math.sin(theta) + rand() * noise * 3;
      label = inner ? 1 : -1;
    } else if (name === 'xor') {
      x1 = rand() * 5;
      x2 = rand() * 5;
      label = x1 * x2 >= 0 ? 1 : -1;
      x1 += rand() * noise * 3;
      x2 += rand() * noise * 3;
    } else if (name === 'gaussian') {
      const positive = i < count / 2;
      const cx = positive ? 2 : -2;
      const cy = positive ? 2 : -2;
      x1 = cx + rand() * (1 + noise * 3);
      x2 = cy + rand() * (1 + noise * 3);
      label = positive ? 1 : -1;
    } else if (name === 'spiral') {
      const positive = i < count / 2;
      const n = (i % (count / 2)) / (count / 2);
      const r = n * 5;
      const t = 1.75 * n * 2 * Math.PI + (positive ? 0 : Math.PI);
      x1 = r * Math.sin(t) + rand() * noise * 3;
      x2 = r * Math.cos(t) + rand() * noise * 3;
      label = positive ? 1 : -1;
    }

    if (opts.problem === 'regression') {
      // Smooth target: sin-based surface so students can see nonlinear fitting
      label = Math.sin(x1 * 0.6) * Math.cos(x2 * 0.6);
    }

    samples.push({ x1, x2, label });
  }

  return samples;
}
