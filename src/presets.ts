import { blankWeights, createNetwork } from "./engine";
import type { ActivationKind, NetworkState } from "./types";

export interface Lab1Variant {
  q21: number;
  q22: number;
  q23: number;
  w14: number;
  w24: number;
  w34: number;
  x1: number;
  x2: number;
}

const LAB1_VARIANT_ROWS: Array<[number, number, number, number, number, number, number, number]> = [
  [0, 0.6, -0.5, 1.0, -0.1, 1.6, 0.1, 0.9],
  [0, 0.5, -0.6, 0.9, -0.2, 1.5, 0.8, -0.3],
  [0, 0.3, -0.4, 0.8, -0.3, 1.4, -0.4, 0.7],
  [0, 0.7, -0.7, 0.7, 0.4, 1.3, 0.6, -0.5],
  [0, 0.4, -0.3, 0.6, -0.5, 1.7, 1.0, 0.7],
  [0, 0.2, -0.8, -0.5, 0.6, 1.8, -0.1, 0.8],
  [0, 0.1, -0.9, -0.4, 0.7, 1.9, -0.2, 0.9],
  [0, 0.8, -0.2, -0.3, 0.8, 1.2, 0.5, -0.6],
  [0, 0.7, -0.6, 0.6, 0.3, 1.3, 0.5, -0.4],
  [0, 0.3, -0.2, 0.7, -0.3, 1.4, -0.3, 0.7],
  [0, 0.6, -0.5, 0.9, -0.1, 1.5, 0.1, 0.8],
  [0, 0.9, -0.1, -0.2, 0.9, 1.1, 0.3, 0.7],
];

export const LAB1_VARIANTS: Lab1Variant[] = LAB1_VARIANT_ROWS.map(([q21, q22, q23, w14, w24, w34, x1, x2]) => ({
  q21,
  q22,
  q23,
  w14,
  w24,
  w34,
  x1,
  x2,
}));

function baseLab1(title: string): NetworkState {
  const state = createNetwork(16);
  state.title = title;
  state.model = "scatter";
  state.weights = blankWeights(16);
  state.graph.selected = [0, 1];
  // Слоты, не участвующие в конкретном опыте ЛР1, остаются видимыми для
  // исходной нумерации, но нулевой линейный выход не подсвечивает их как
  // сработавшие только из-за равенства P = Q = 0.
  state.neurons.forEach((neuron) => { neuron.activation = "linear"; });
  return state;
}

export function lab1SingleNeuronPreset(
  activation: ActivationKind,
  threshold = 0,
): NetworkState {
  const labels: Record<ActivationKind, string> = {
    "hard-limit": "знаковая функция",
    linear: "линейная функция",
    sigmoid: "сигмоидальная функция",
  };
  const state = baseLab1(`ЛР1: ${labels[activation]}`);
  const input = state.neurons[0]!;
  input.role = "input";
  input.activation = "linear";
  input.inputKind = "harmonic";
  input.harmonic = { amplitude: 1, omega: 1, phase: 0, deltaTime: 1 };

  const output = state.neurons[1]!;
  output.role = "output";
  output.activation = activation;
  output.threshold = threshold;
  state.weights[0]![1] = 1;
  return state;
}

export function lab1SummingPreset(
  variantNumber = 1,
  activation: ActivationKind = "hard-limit",
  threshold = 0,
): NetworkState {
  const variant = LAB1_VARIANTS[Math.max(0, Math.min(11, variantNumber - 1))]!;
  const state = baseLab1(`ЛР1: суммирующий нейрон, вариант ${variantNumber}`);
  state.graph.selected = [0, 1, 2, 3];

  const inputs = state.neurons.slice(0, 3);
  inputs.forEach((neuron, index) => {
    neuron.role = "input";
    neuron.activation = "linear";
    neuron.threshold = [variant.q21, variant.q22, variant.q23][index]!;
  });
  inputs[0]!.constantInput = variant.x1;
  inputs[1]!.constantInput = variant.x2;
  inputs[2]!.inputKind = "harmonic";
  inputs[2]!.harmonic = { amplitude: 1, omega: 1, phase: 0, deltaTime: 1 };

  const output = state.neurons[3]!;
  output.role = "output";
  output.activation = activation;
  output.threshold = threshold;
  state.weights[0]![3] = variant.w14;
  state.weights[1]![3] = variant.w24;
  state.weights[2]![3] = variant.w34;
  return state;
}

export function lab2PerceptronPreset(): NetworkState {
  const state = createNetwork(15);
  state.title = "ЛР2: чётные и нечётные цифры";
  state.model = "perceptron";
  state.weights = blankWeights(15);
  state.perceptron = {
    inputs: 14,
    outputs: 1,
    trainableInputs: Array.from({ length: 14 }, (_, index) => index),
    learningRate: 0.1,
    desired: [1],
    trained: false,
    trainingSteps: 0,
    lastError: 0,
  };
  state.neurons.forEach((neuron, index) => {
    neuron.role = index < 14 ? "input" : "output";
    neuron.inputKind = "constant";
    neuron.activation = index < 14 ? "linear" : "hard-limit";
    if (index < 14) neuron.linear = { slope: 1, intercept: 0 };
    neuron.threshold = 0;
  });
  // Небольшие ненулевые стартовые веса, как рекомендует ЛР2.
  for (let source = 0; source < 12; source += 1) {
    state.weights[source]![14] = source % 2 === 0 ? 0.05 : -0.05;
  }
  // Нормативно приведённый в методичке образ цифры 1: {010110010010}.
  "010110010010".split("").forEach((bit, index) => {
    state.neurons[index]!.constantInput = Number(bit);
  });
  state.graph.selected = [14];
  return state;
}
