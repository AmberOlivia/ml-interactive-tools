export type FeatureName = 'x1' | 'x2' | 'x3' | 'x4' | 'x5';
export type ConstantFeatureName = 'x3' | 'x4' | 'x5';

export interface Feature {
  name: FeatureName;
  label: string;
  // True for x1/x2 (data-driven); false for x3/x4/x5 (user-set constant)
  isDataDriven: boolean;
}

export const FEATURES: Record<FeatureName, Feature> = {
  x1: { name: 'x1', label: 'x₁', isDataDriven: true },
  x2: { name: 'x2', label: 'x₂', isDataDriven: true },
  x3: { name: 'x3', label: 'x₃', isDataDriven: false },
  x4: { name: 'x4', label: 'x₄', isDataDriven: false },
  x5: { name: 'x5', label: 'x₅', isDataDriven: false },
};

export const FEATURE_NAMES: FeatureName[] = ['x1', 'x2', 'x3', 'x4', 'x5'];
export const CONSTANT_FEATURE_NAMES: ConstantFeatureName[] = ['x3', 'x4', 'x5'];

export type ConstantValues = Record<ConstantFeatureName, number>;

export function computeFeatures(
  x1: number,
  x2: number,
  selected: FeatureName[],
  constants: ConstantValues,
): number[] {
  return selected.map((n) => {
    if (n === 'x1') return x1;
    if (n === 'x2') return x2;
    return constants[n];
  });
}
