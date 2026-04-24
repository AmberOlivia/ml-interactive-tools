export type FeatureName =
  | 'x0'
  | 'x1'
  | 'x2'
  | 'x1Squared'
  | 'x2Squared'
  | 'x1x2'
  | 'sinX1'
  | 'sinX2';

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
  x1Squared: { name: 'x1Squared', label: 'x₁²', isDataDriven: true },
  x2Squared: { name: 'x2Squared', label: 'x₂²', isDataDriven: true },
  x1x2: { name: 'x1x2', label: 'x₁x₂', isDataDriven: true },
  sinX1: { name: 'sinX1', label: 'sin(x₁)', isDataDriven: true },
  sinX2: { name: 'sinX2', label: 'sin(x₂)', isDataDriven: true },
};

export const FEATURE_NAMES: FeatureName[] = [
  'x0',
  'x1',
  'x2',
  'x1Squared',
  'x2Squared',
  'x1x2',
  'sinX1',
  'sinX2',
];

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
      case 'x1Squared':
        return x1 * x1;
      case 'x2Squared':
        return x2 * x2;
      case 'x1x2':
        return x1 * x2;
      case 'sinX1':
        return Math.sin(x1);
      case 'sinX2':
        return Math.sin(x2);
    }
  });
}
