export type FeatureName = 'x0' | 'x1' | 'x2';

// Constant inputs that don't come from the dataset.
export type ConstantFeatureName = 'x0';

export interface Feature {
  name: FeatureName;
  label: string;
  // True: derived from (x1, x2). False: constant input set outside the dataset.
  isDataDriven: boolean;
}

export const FEATURES: Record<FeatureName, Feature> = {
  x0: { name: 'x0', label: 'x₀ = 1 (bias)', isDataDriven: false },
  x1: { name: 'x1', label: 'x₁', isDataDriven: true },
  x2: { name: 'x2', label: 'x₂', isDataDriven: true },
};

export const FEATURE_NAMES: FeatureName[] = ['x0', 'x1', 'x2'];

export const CONSTANT_FEATURE_NAMES: ConstantFeatureName[] = ['x0'];

export type ConstantValues = Record<ConstantFeatureName, number>;

export function computeFeatures(
  x1: number,
  x2: number,
  selected: FeatureName[],
  constants: ConstantValues,
): number[] {
  return selected.map((n) => {
    switch (n) {
      case 'x0':
        return constants.x0;
      case 'x1':
        return x1;
      case 'x2':
        return x2;
    }
  });
}
