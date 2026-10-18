export type ModelKind = "scatter" | "perceptron";
export type NeuronRole = "input" | "ordinary" | "output";
export type ActivationKind = "hard-limit" | "linear" | "sigmoid";
export type InputKind = "constant" | "harmonic";
export type GraphKind = "lines" | "bars";

export interface LinearParameters {
  slope: number;
  intercept: number;
}

export interface HarmonicParameters {
  amplitude: number;
  omega: number;
  phase: number;
  deltaTime: number;
}

export interface Neuron {
  id: number;
  role: NeuronRole;
  threshold: number;
  activation: ActivationKind;
  linear: LinearParameters;
  inputKind: InputKind;
  constantInput: number;
  harmonic: HarmonicParameters;
  potential: number;
  output: number;
}

export interface PerceptronArchitecture {
  inputs: number;
  outputs: number;
  /** @deprecated Kept for project-format compatibility; all inputs are trained. */
  trainableInputs: number[];
  learningRate: number;
  desired: number[];
  trained: boolean;
  trainingSteps: number;
  lastError: number;
  trainingExamples?: TrainingExample[];
  patternColumns?: number;
}

export interface TrainingExample {
  name: string;
  inputs: number[];
  desired: number[];
}

export interface ExamplesReport {
  rows: { outputs: number[]; potentials: number[]; correct: boolean }[];
  correct: number;
  total: number;
}

export interface EpochTrainingResult {
  state: NetworkState;
  report: ExamplesReport | null;
  changedWeights: number;
  blockedReason?: string;
}

export interface GraphSample {
  step: number;
  time: number;
  potentials: number[];
  outputs: number[];
}

export interface NetworkState {
  schemaVersion: 2;
  title: string;
  model: ModelKind;
  neurons: Neuron[];
  weights: number[][];
  perceptron: PerceptronArchitecture;
  step: number;
  time: number;
  history: GraphSample[];
  graph: {
    selected: number[];
    kind: GraphKind;
    maxSamples: number;
  };
}

export interface StepResult {
  state: NetworkState;
  outputs: number[];
}

export interface TrainingResult {
  state: NetworkState;
  learned: boolean;
  error: number;
  changedWeights: number;
  blockedReason?: string;
}
