import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import type { NetworkState, TrainingExample } from "../src/types";
import {
  applyActivation,
  configurePerceptron,
  createNetwork,
  createNeuron,
  deserializeState,
  evaluateExamples,
  MAX_TRAINING_EXAMPLES,
  resetRuntime,
  resizeNetwork,
  runStep,
  serializeState,
  trainStep,
  trainEpoch,
  switchModel,
  trainingSupportIssue,
  trainingExamplesIssue,
  validateState,
  zeroInputs,
} from "../src/engine";
import {
  lab1SingleNeuronPreset,
  lab1SummingPreset,
  lab2PerceptronPreset,
} from "../src/presets";

describe("начальные параметры гармонического входа", () => {
  const defaults = { amplitude: 1, omega: 1, phase: 1, deltaTime: 0.01 };
  const labParameters = { amplitude: 1, omega: 1, phase: 0, deltaTime: 1 };

  it("задаёт новые параметры каждому создаваемому нейрону", () => {
    for (let id = 1; id <= 32; id++) {
      const neuron = createNeuron(id);
      expect(neuron.id).toBe(id);
      expect(neuron.harmonic).toEqual(defaults);
      expect(neuron.inputKind).toBe("constant");
    }
    const first = createNeuron(1), second = createNeuron(2);
    first.harmonic.phase = 9;
    expect(second.harmonic).toEqual(defaults);
  });

  it.each([2, 16, 32])("создаёт сеть из %i нейронов с новыми значениями по умолчанию", (count) => {
    const state = createNetwork(count);
    expect(state.neurons).toHaveLength(count);
    expect(state.neurons.map(neuron => neuron.harmonic)).toEqual(Array.from({ length: count }, () => defaults));
  });

  it("использует новые значения только для добавленных нейронов, не переписывая существующие", () => {
    const source = createNetwork(2);
    source.neurons[0]!.harmonic = { amplitude: 2.5, omega: 0.75, phase: -0.4, deltaTime: 0.125 };
    source.neurons[1]!.harmonic = { ...labParameters };
    const before = structuredClone(source);
    for (const expanded of [resizeNetwork(source, 8), configurePerceptron(source, 6, 2)]) {
      expect(expanded.neurons.slice(0, 2).map(neuron => neuron.harmonic)).toEqual(before.neurons.map(neuron => neuron.harmonic));
      expect(expanded.neurons.slice(2).map(neuron => neuron.harmonic)).toEqual(Array.from({ length: 6 }, () => defaults));
    }
    expect(source).toEqual(before);
  });

  it("считает первые P = cos(1 + 0,01·n) и время 0,01·n после включения гармонического входа", () => {
    let state = createNetwork(2);
    state.neurons[0]!.role = "input";
    state.neurons[0]!.activation = "linear";
    state.neurons[0]!.inputKind = "harmonic";
    for (let step = 1; step <= 7; step++) {
      state = runStep(state).state;
      const time = 0.01 * step;
      expect(state.step).toBe(step);
      expect(state.time).toBeCloseTo(time, 14);
      expect(state.neurons[0]!.potential).toBeCloseTo(Math.cos(1 + time), 14);
      expect(state.neurons[0]!.output).toBeCloseTo(Math.cos(1 + time), 14);
      expect(state.history.at(-1)!.time).toBeCloseTo(time, 14);
      expect(state.history.at(-1)!.outputs[0]).toBeCloseTo(Math.cos(1 + time), 14);
    }
  });

  it("после ручного ввода фазы 0 и шага 1 воспроизводит опыт ЛР1 без изменения формул", () => {
    let manual = createNetwork(2);
    manual.neurons[0]!.role = "input";
    manual.neurons[0]!.activation = "linear";
    manual.neurons[0]!.inputKind = "harmonic";
    manual.neurons[0]!.harmonic.phase = 0;
    manual.neurons[0]!.harmonic.deltaTime = 1;
    manual.neurons[1]!.role = "output";
    manual.neurons[1]!.activation = "hard-limit";
    manual.neurons[1]!.threshold = 0.5;
    manual.weights[0]![1] = 1;
    let preset = lab1SingleNeuronPreset("hard-limit", 0.5);
    const expectedOutputs = [1, 0, 0, 0, 0, 1, 1];
    for (let step = 1; step <= 7; step++) {
      manual = runStep(manual).state;
      preset = runStep(preset).state;
      expect(manual.time).toBe(step);
      expect(manual.neurons[0]!.potential).toBeCloseTo(Math.cos(step), 14);
      expect(manual.neurons[1]!.output).toBe(expectedOutputs[step - 1]);
      expect(manual.neurons.map(neuron => [neuron.potential, neuron.output])).toEqual(preset.neurons.slice(0, 2).map(neuron => [neuron.potential, neuron.output]));
    }
  });

  it.each([
    { name: "прежние 0/1", parameters: labParameters },
    { name: "пользовательские", parameters: { amplitude: 2.5, omega: 0.75, phase: -1.2, deltaTime: 0.125 } },
  ])("сохраняет и импортирует $name параметры, не заменяя их новыми defaults", ({ parameters }) => {
    let state = createNetwork(2);
    state.neurons.forEach(neuron => { neuron.harmonic = { ...parameters }; });
    state.neurons[0]!.role = "input";
    state.neurons[0]!.activation = "linear";
    state.neurons[0]!.inputKind = "harmonic";
    state = runStep(state).state;
    const before = structuredClone(state);
    const saved = JSON.parse(serializeState(state)) as NetworkState;
    const loaded = deserializeState(saved).state;
    for (const candidate of [saved, loaded]) {
      expect(candidate.neurons.map(neuron => neuron.harmonic)).toEqual([parameters, parameters]);
    }
    const first = runStep(loaded).state;
    expect(first.time).toBe(parameters.deltaTime);
    expect(first.neurons[0]!.potential).toBeCloseTo(parameters.amplitude * Math.cos(parameters.omega * parameters.deltaTime + parameters.phase), 14);
    expect(state).toEqual(before);
  });

  it("оставляет активные гармонические входы всех пресетов ЛР1 с фазой 0 и шагом 1", () => {
    for (const activation of ["hard-limit", "linear", "sigmoid"] as const) {
      const presets = [lab1SingleNeuronPreset(activation), ...Array.from({ length: 12 }, (_, index) => lab1SummingPreset(index + 1, activation))];
      for (const preset of presets) {
        const inputs = preset.neurons.filter(neuron => neuron.role === "input" && neuron.inputKind === "harmonic");
        expect(inputs).toHaveLength(1);
        expect(inputs[0]!.harmonic).toEqual(labParameters);
        expect(runStep(preset).state.time).toBe(1);
      }
    }
    const lab2 = lab2PerceptronPreset();
    expect(lab2.neurons.every(neuron => neuron.inputKind === "constant")).toBe(true);
    expect(runStep(lab2).state.time).toBe(1);
  });
});

describe("функции активации", () => {
  it("сохраняет строгую отсечку у знаковой и линейной функций", () => {
    for (const kind of ["hard-limit", "linear"] as const) {
      expect(applyActivation(kind, 0.5, 0.5, { slope: 2, intercept: 10 })).toBe(0);
      expect(applyActivation(kind, 0.49, 0.5, { slope: 2, intercept: 10 })).toBe(0);
    }
    expect(applyActivation("hard-limit", 0.500001, 0.5)).toBe(1);
    expect(applyActivation("hard-limit", 0.49, 0.5)).toBe(0);
  });

  it("вычитает порог и применяет подтверждённую EXE формулу k·X+b", () => {
    const params = { slope: 2, intercept: 1 };
    expect(applyActivation("linear", 4, 0.5, params)).toBe(8);
    expect(applyActivation("linear", -2, 0.5, params)).toBe(0);
    // The gate applies to P-Q, not to the final k*(P-Q)+b.
    expect(applyActivation("linear", 2, 0.5, { slope: -2, intercept: 1 })).toBe(-2);
  });

  it("использует полную логистику 1/(1+exp(-(P-Q))) без отсечки", () => {
    expect(applyActivation("sigmoid", 0, 0.5)).toBeCloseTo(0.377540669, 9);
    expect(applyActivation("sigmoid", 0.5, 0.5)).toBe(0.5);
    expect(applyActivation("sigmoid", 1, 0.5)).toBeCloseTo(0.622459331, 9);
    expect(applyActivation("sigmoid", 1.5, 0.5)).toBeCloseTo(0.731058579, 9);
  });

  it.each([-5, -.4, 0, .4, 5])("сдвигает непрерывную сигмоиду порогом Q=%s", threshold => {
    expect(applyActivation("sigmoid", threshold, threshold)).toBe(.5);
    for (const offset of [-8, -2, -1, -.001, .001, 1, 2, 8]) {
      const actual = applyActivation("sigmoid", threshold + offset, threshold);
      expect(actual).toBeCloseTo(1 / (1 + Math.exp(-offset)), 14);
      expect(actual).toBeGreaterThan(0);
      expect(actual).toBeLessThan(1);
      expect(actual + applyActivation("sigmoid", threshold - offset, threshold)).toBeCloseTo(1, 14);
    }
  });

  it("не создаёт NaN/Infinity для крайних конечных входов сигмоиды", () => {
    for (const potential of [-Number.MAX_VALUE, -1000, 1000, Number.MAX_VALUE]) {
      for (const threshold of [-Number.MAX_VALUE, 0, Number.MAX_VALUE]) {
        const actual = applyActivation("sigmoid", potential, threshold);
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBeGreaterThanOrEqual(0);
        expect(actual).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe("обнуление входов начинает новый опыт, не меняя схему", () => {
  it.each(["scatter", "perceptron"] as const)("сохраняет параметры и веса модели %s", model => {
    let state = model === "scatter" ? createNetwork(3) : configurePerceptron(createNetwork(), 2, 1);
    state.neurons[0]!.role = "input";
    state.neurons[0]!.activation = "sigmoid";
    state.neurons[0]!.threshold = -.4;
    state.neurons[0]!.inputKind = "harmonic";
    state.neurons[0]!.harmonic = { amplitude: 2, omega: .7, phase: -.3, deltaTime: .125 };
    state.neurons[0]!.constantInput = 2;
    state.neurons[1]!.role = "input";
    state.neurons[1]!.constantInput = 3;
    state.neurons[2]!.role = "output";
    state.neurons[2]!.activation = "sigmoid";
    state.neurons[2]!.threshold = .6;
    state.weights[0]![2] = -1.25;
    state.weights[1]![2] = .75;
    for (let i = 0; i < 4; i++) state = runStep(state).state;
    state.perceptron.trainingExamples = [{ name: "Сохранённый образ", inputs: [1, 0], desired: [1] }];
    const before = structuredClone(state);
    const result = zeroInputs(state);
    expect(result.step).toBe(0);
    expect(result.time).toBe(0);
    expect(result.history).toEqual([]);
    expect(result.weights).toEqual(before.weights);
    expect(result.graph).toEqual(before.graph);
    expect(result.model).toBe(before.model);
    expect(result.perceptron.trainingExamples).toEqual(before.perceptron.trainingExamples);
    expect(result.neurons).toEqual(before.neurons.map(neuron => ({ ...neuron, potential: 0, output: 0, constantInput: neuron.role === "input" ? 0 : neuron.constantInput })));
    expect(state).toEqual(before);
    const next = runStep(result).state;
    expect(next.step).toBe(1);
    expect(next.neurons[0]!.potential).toBeCloseTo(2 * Math.cos(.7 * .125 - .3), 14);
    expect(next.neurons[1]!.potential).toBe(0);
    expect(next.neurons[0]!.output).toBeGreaterThan(0);
    expect(next.history).toHaveLength(1);
  });
});

describe("сценарии ЛР1", () => {
  it("воспроизводит семь значений cos(k) и hard-limit для варианта 1", () => {
    const expected: Record<number, number[]> = {
      0: [1, 0, 0, 0, 1, 1, 1],
      0.6: [0, 0, 0, 0, 0, 1, 1],
      [-0.5]: [1, 1, 1, 1, 1, 1, 1],
    };
    for (const [thresholdText, outputs] of Object.entries(expected)) {
      let state = lab1SingleNeuronPreset("hard-limit", Number(thresholdText));
      const actual: number[] = [];
      for (let step = 0; step < 7; step += 1) {
        state = runStep(state).state;
        actual.push(state.neurons[1]!.output);
      }
      expect(actual).toEqual(outputs);
    }
  });

  it("считает потенциал сумматора варианта 1 без промежуточного округления", () => {
    let state = lab1SummingPreset(1, "hard-limit", 0);
    state = runStep(state).state;
    expect(state.neurons[3]!.potential).toBeCloseTo(1.734483689, 9);
    for (let step = 1; step < 5; step += 1) state = runStep(state).state;
    expect(state.neurons[3]!.potential).toBeCloseTo(1.323859497, 9);
    expect(state.neurons[3]!.output).toBe(1);
  });

  it("оставляет нулевыми неиспользуемые слоты H3–H16 в пресете одиночного нейрона", () => {
    const state = runStep(lab1SingleNeuronPreset("hard-limit", 0)).state;
    expect(state.neurons.slice(2).every((neuron) => neuron.output === 0)).toBe(true);
  });

  it("передаёт текущий выход H1 в потенциал H2 с весом 1 на каждом шаге", () => {
    let state = lab1SingleNeuronPreset("hard-limit", 0.2);
    for (let step = 1; step <= 19; step += 1) {
      state = runStep(state).state;
      expect(state.neurons[0]!.potential).toBeCloseTo(Math.cos(step), 12);
      expect(state.neurons[0]!.output).toBeCloseTo(Math.max(0, Math.cos(step)), 12);
      expect(state.neurons[1]!.potential).toBe(state.neurons[0]!.output);
    }
  });

  it("воспроизводит семь потенциалов сумматора варианта 2 из отчёта", () => {
    let state = lab1SummingPreset(2, "linear", 0.5);
    const expected = [2.430453459, 0.995779745, 0.72, 0.72, 2.045493278, 3.060255430, 2.750853382];
    expect(state.neurons.slice(0, 3).map((neuron) => neuron.threshold)).toEqual([0, 0.5, -0.6]);
    for (const potential of expected) {
      state = runStep(state).state;
      expect(state.neurons[3]!.potential).toBeCloseTo(potential, 9);
      expect(state.neurons[3]!.output).toBeCloseTo(potential - 0.5, 9);
    }
  });
});

describe("динамическая сеть", () => {
  it("сохраняет ведущую подматрицу и нейроны при увеличении N", () => {
    const source = createNetwork(4);
    source.neurons[0]!.threshold = 0.25;
    source.weights[0]![3] = 0.75;
    const resized = resizeNetwork(source, 8);
    expect(resized.neurons).toHaveLength(8);
    expect(resized.neurons[0]!.threshold).toBe(0.25);
    expect(resized.weights[0]![3]).toBe(0.75);
    expect(resized.weights[7]![7]).toBe(0);
    expect(resized.history).toEqual([]);
  });

  it("экспортирует минимальную свободную сеть из двух нейронов", () => {
    const state = createNetwork(2);
    expect(state.perceptron).not.toHaveProperty("ordinary");
    expect(validateState(JSON.parse(serializeState(state)))).toBe(true);
  });

  it("безопасно обрабатывает NaN и бесконечность в публичном изменении размера", () => {
    expect(createNetwork(Number.NaN).neurons).toHaveLength(16);
    const source = createNetwork(7);
    expect(resizeNetwork(source, Number.POSITIVE_INFINITY).neurons).toHaveLength(7);
    expect(resizeNetwork(source, Number.NEGATIVE_INFINITY).neurons).toHaveLength(7);
  });

  it("обрабатывает обратную связь через выход прошлого шага", () => {
    let state = createNetwork(2);
    state.neurons[0]!.role = "input";
    state.neurons[0]!.activation = "linear";
    state.neurons[0]!.linear = { slope: 1, intercept: 0 };
    state.neurons[0]!.constantInput = 1;
    state.neurons[1]!.role = "output";
    state.neurons[1]!.activation = "linear";
    state.neurons[1]!.linear = { slope: 1, intercept: 0 };
    state.weights[0]![1] = 1;
    state.weights[1]![1] = 1;
    state = runStep(state).state;
    expect(state.neurons[1]!.output).toBe(1);
    state = runStep(state).state;
    expect(state.neurons[1]!.output).toBe(2);
  });

  it("отклоняет переполнение расчёта, сохраняя последнее состояние", () => {
    const state = createNetwork(2);
    state.neurons[0]!.role = "input";
    state.neurons[0]!.activation = "linear";
    state.neurons[0]!.constantInput = 2;
    state.weights[0]![1] = Number.MAX_VALUE;
    const snapshot = structuredClone(state);
    expect(() => runStep(state)).toThrowError(/Слишком большие значения/);
    expect(state).toEqual(snapshot);
    expect(validateState(state)).toBe(true);
  });
});

interface DigitExample { digit: number; bits: string; target: number }
const oracle = JSON.parse(readFileSync(new URL("./fixtures/digits.json", import.meta.url), "utf8")) as {
  lr2: { dataset: DigitExample[]; reportRawBitsDifferentCoordinateSet: string[] };
};
const digits = oracle.lr2.dataset;

function enterBits(state: NetworkState, bits: string): void {
  [...bits].forEach((bit, index) => { state.neurons[index]!.constantInput = Number(bit); });
}

function classifyDigits(state: NetworkState, examples = digits): number[][] {
  return examples.map(({ bits }) => {
    const entered = structuredClone(state);
    enterBits(entered, bits);
    return runStep(entered).state.neurons.slice(state.perceptron.inputs).map(neuron => neuron.output);
  });
}

function learnDigits(initial: NetworkState, examples: DigitExample[], targets: number[][]): NetworkState {
  let state = initial;
  for (let epoch = 0; epoch < 500; epoch += 1) {
    for (let i = 0; i < examples.length; i += 1) {
      enterBits(state, examples[i]!.bits);
      state.perceptron.desired = [...targets[i]!];
      for (let attempt = 0; attempt < 100; attempt += 1) {
        const result = trainStep(state);
        expect(result.blockedReason).toBeUndefined();
        state = result.state;
        if (result.learned) break;
      }
    }
    if (classifyDigits(state, examples).every((actual, i) =>
      actual.every((value, j) => value === targets[i]![j]))) return state;
  }
  throw new Error("Выборка не разделена после 500 полных проходов");
}

type LegacyFixture = Omit<NetworkState, "schemaVersion" | "perceptron"> & {
  schemaVersion: 1;
  perceptron: NetworkState["perceptron"] & { ordinary: number };
};

function legacyFixture(): LegacyFixture {
  return JSON.parse(readFileSync(new URL("./fixtures/legacy-lr2-network.json", import.meta.url), "utf8")) as LegacyFixture;
}

describe("прямой перцептрон", () => {
  it("содержит только входы и выходы, с нулевой матрицей и стандартными порогами", () => {
    const state = configurePerceptron(createNetwork(), 14, 2);
    expect(state.schemaVersion).toBe(2);
    expect(state.neurons).toHaveLength(16);
    expect(state.neurons.slice(0, 14).every(neuron => neuron.role === "input" && neuron.activation === "linear")).toBe(true);
    expect(state.neurons.slice(14).map(neuron => neuron.role)).toEqual(["output", "output"]);
    expect(state.neurons.every(neuron => neuron.threshold === 0)).toBe(true);
    expect(state.weights.flat().every(weight => weight === 0)).toBe(true);
    expect(state.perceptron).not.toHaveProperty("ordinary");
    expect(validateState(state)).toBe(true);
  });

  it.each([[1, 1], [31, 1], [1, 31], [14, 2]])("поддерживает %i входов и %i выходов в пределах 2–32", (inputs, outputs) => {
    const state = configurePerceptron(createNetwork(), inputs, outputs);
    expect(state.neurons).toHaveLength(inputs + outputs);
    expect(validateState(state)).toBe(true);
    expect(trainingSupportIssue(state)).toBeNull();
  });

  it("переключает свободную сеть на прямые входы/выходы без скрытого остатка", () => {
    const state = switchModel(createNetwork(), "perceptron");
    expect(state.perceptron.inputs).toBe(14);
    expect(state.perceptron.outputs).toBe(1);
    expect(state.neurons).toHaveLength(15);
    expect(validateState(state)).toBe(true);
    expect(validateState(switchModel(state, "scatter"))).toBe(true);
  });

  it("не переносит пороги старых ролей при новой топологии", () => {
    let state = configurePerceptron(createNetwork(), 14, 2);
    state.neurons[0]!.threshold = 0.8;
    state.neurons[14]!.threshold = 0.7;
    state.weights[1]![14] = 0.4;
    state = configurePerceptron(state, 2, 2);
    expect(state.neurons.map(neuron => neuron.role)).toEqual(["input", "input", "output", "output"]);
    expect(state.neurons.map(neuron => neuron.threshold)).toEqual([0, 0, 0, 0]);
    expect(state.weights.flat().every(weight => weight === 0)).toBe(true);
    expect(validateState(resizeNetwork(state, 2))).toBe(true);
    expect(resizeNetwork(state, 2).neurons).toHaveLength(2);
  });

  it("считает прямые взвешенные суммы обоих выходов за один шаг", () => {
    const state = configurePerceptron(createNetwork(), 2, 2);
    state.neurons[0]!.constantInput = 1;
    state.neurons[1]!.constantInput = 0.5;
    state.weights[0]![2] = 0.8;
    state.weights[1]![2] = -0.2;
    state.weights[0]![3] = -0.4;
    state.weights[1]![3] = 0.1;
    const result = runStep(state).state;
    expect(result.neurons[2]!.potential).toBeCloseTo(0.7, 12);
    expect(result.neurons[3]!.potential).toBeCloseTo(-0.35, 12);
    expect(result.neurons.slice(2).map(neuron => neuron.output)).toEqual([1, 0]);
  });

  it("не меняет алгоритм при добавлении второго выхода", () => {
    const single = configurePerceptron(createNetwork(), 2, 1);
    const multiple = configurePerceptron(createNetwork(), 2, 2);
    for (const state of [single, multiple]) {
      state.neurons[0]!.constantInput = 1;
      state.neurons[1]!.constantInput = 1;
      state.neurons[2]!.threshold = 0.5;
      state.weights[0]![2] = 0.2;
      state.weights[1]![2] = 0.2;
    }
    multiple.perceptron.desired = [1, 0];
    const a = trainStep(single);
    const b = trainStep(multiple);
    expect(a.learned).toBe(true);
    expect(b.learned).toBe(true);
    for (const source of [0, 1]) {
      expect(a.state.weights[source]![2]).toBeCloseTo(0.3, 12);
      expect(b.state.weights[source]![2]).toBe(a.state.weights[source]![2]);
      expect(b.state.weights[source]![3]).toBe(0);
    }
    expect(b.state.neurons[2]!.threshold).toBe(0.5);
  });

  it("каждый выход обучает только собственные входящие веса", () => {
    let state = configurePerceptron(createNetwork(), 2, 2);
    state.neurons[0]!.constantInput = 1;
    state.neurons[1]!.constantInput = 0;
    state.perceptron.desired = [1, 0];
    const first = trainStep(state);
    expect(first.changedWeights).toBe(1);
    expect(first.state.weights[0]![2]).toBe(0.1);
    expect(first.state.weights[0]![3]).toBe(0);
    expect(first.state.neurons.slice(2).map(neuron => neuron.output)).toEqual([1, 0]);
    state = first.state;
    state.neurons[0]!.constantInput = 0;
    state.neurons[1]!.constantInput = 1;
    state.perceptron.desired = [0, 1];
    const second = trainStep(state);
    expect(second.changedWeights).toBe(1);
    expect(second.state.weights[0]![2]).toBe(0.1);
    expect(second.state.weights[1]![2]).toBe(0);
    expect(second.state.weights[1]![3]).toBe(0.1);
    expect(second.state.neurons.slice(2).map(neuron => neuron.output)).toEqual([0, 1]);
  });

  it("считает общую ошибку и корректирует все ошибочные выходы", () => {
    const state = configurePerceptron(createNetwork(), 1, 2);
    state.neurons[0]!.constantInput = 1;
    state.weights[0]![1] = -0.1;
    state.weights[0]![2] = 0.1;
    state.perceptron.learningRate = 0.2;
    state.perceptron.desired = [1, 0];
    const result = trainStep(state);
    expect(result.error).toBe(1);
    expect(result.changedWeights).toBe(2);
    expect(result.learned).toBe(true);
    expect(result.state.weights[0]!.slice(1)).toEqual([0.1, -0.1]);
  });

  it("обучает все 14 входов ЛР2, включая ненулевые H13/H14", () => {
    const state = lab2PerceptronPreset();
    expect(state.neurons).toHaveLength(15);
    expect(state.neurons[14]!.role).toBe("output");
    expect(state.neurons[14]!.threshold).toBe(0);
    expect(state.perceptron.trainableInputs).toEqual(Array.from({ length: 14 }, (_, index) => index));
    state.neurons.slice(0, 12).forEach(neuron => { neuron.constantInput = 0; });
    state.neurons[12]!.constantInput = 1;
    state.neurons[13]!.constantInput = 1;
    state.weights[12]![14] = 0.4;
    state.weights[13]![14] = 0.4;
    state.perceptron.desired = [0];
    const result = trainStep(state);
    expect(result.state.weights[12]![14]).toBeCloseTo(0.3, 12);
    expect(result.state.weights[13]![14]).toBeCloseTo(0.3, 12);
    expect(result.changedWeights).toBe(2);
    expect(result.learned).toBe(false);
    expect(result.blockedReason).toBeUndefined();
  });

  it("игнорирует старую маску: ненулевой вход обучает оба выхода, нулевые выходы входных нейронов дают нулевую поправку", () => {
    const state = configurePerceptron(createNetwork(), 3, 2);
    state.perceptron.trainableInputs = [0];
    state.perceptron.learningRate = 0.2;
    state.perceptron.desired = [1, 0];
    state.neurons[0]!.constantInput = 2;
    state.neurons[0]!.threshold = 3; // Raw input is nonzero, but its output is zero.
    state.neurons[1]!.constantInput = 1;
    state.neurons[2]!.constantInput = 0;
    state.weights[0]![3] = 7;
    state.weights[0]![4] = -4;
    state.weights[1]![3] = -0.1;
    state.weights[1]![4] = 0.1;
    state.weights[2]![3] = 0.5;
    state.weights[2]![4] = 0.6;
    const before = structuredClone(state);
    const result = trainStep(state);
    expect(result.blockedReason).toBeUndefined();
    expect(result.learned).toBe(true);
    expect(result.changedWeights).toBe(2);
    expect(result.state.neurons.slice(0, 3).map(neuron => neuron.output)).toEqual([0, 1, 0]);
    expect(result.state.weights[0]).toEqual(before.weights[0]);
    expect(result.state.weights[2]).toEqual(before.weights[2]);
    expect(result.state.weights[1]!.slice(3)).toEqual([0.1, -0.1]);
    expect(result.state.perceptron.trainableInputs).toEqual([0, 1, 2]);
    expect(state).toEqual(before);
  });

  it("сохраняет время, график и обученные веса при следующих шагах и обучении", () => {
    let state = configurePerceptron(createNetwork(), 1, 1);
    state.neurons[0]!.constantInput = 1;
    state.weights[0]![1] = 0.2;
    state = runStep(state).state;
    state.neurons[1]!.threshold = 0.3;
    const result = trainStep(state);
    expect(result.state.step).toBe(1);
    expect(result.state.time).toBe(1);
    expect(result.state.history).toEqual(state.history);
    const next = runStep(result.state).state;
    expect(next.step).toBe(2);
    expect(next.history).toHaveLength(2);
    const imported = deserializeState(JSON.parse(serializeState(next))).state;
    expect(imported.weights).toEqual(next.weights);
    expect(imported.step).toBe(0);
    expect(imported.history).toEqual([]);
  });

  it.each(["нормативные карты", "векторы отчёта"])("обучает ЛР2 до 10/10 и сохраняет результат: %s", (label) => {
    const examples = label === "нормативные карты" ? digits : digits.map((sample, i) => ({
      ...sample, bits: oracle.lr2.reportRawBitsDifferentCoordinateSet[i]!,
    }));
    const targets = examples.map(sample => [sample.target]);
    const state = learnDigits(lab2PerceptronPreset(), examples, targets);
    expect(classifyDigits(state, examples)).toEqual(targets);
    const loaded = deserializeState(JSON.parse(serializeState(state))).state;
    expect(classifyDigits(loaded, examples)).toEqual(targets);
    expect(loaded.weights[12]![14]).toBe(0);
    expect(loaded.weights[13]![14]).toBe(0);
  });

  it("обучает два независимых признака десяти цифр: нечётность и ≥8", () => {
    let state = configurePerceptron(createNetwork(), 14, 2);
    state.perceptron.trainableInputs = Array.from({ length: 12 }, (_, index) => index);
    for (let source = 0; source < 12; source += 1) {
      state.weights[source]![14] = source % 2 === 0 ? 0.05 : -0.05;
      state.weights[source]![15] = source % 2 === 0 ? 0.05 : -0.05;
    }
    const initial = structuredClone(state.weights);
    const targets = digits.map(sample => [sample.target, sample.digit >= 8 ? 1 : 0]);
    state = learnDigits(state, digits, targets);
    expect(classifyDigits(state)).toEqual(targets);
    for (const output of [14, 15]) {
      expect(state.weights.some((row, source) => row[output] !== initial[source]![output])).toBe(true);
      expect(state.neurons[output]!.threshold).toBe(0);
    }
    expect(state.weights.every((row, source) => row.every((value, target) =>
      value === 0 || (source < 12 && target >= 14)))).toBe(true);
    const imported = deserializeState(JSON.parse(serializeState(state))).state;
    expect(classifyDigits(imported)).toEqual(targets);
  });
});

describe("явный учебный набор и полные проходы", () => {
  const simple: TrainingExample[] = [
    { name: "A", inputs: [1, 0], desired: [1] },
    { name: "B", inputs: [1, 1], desired: [0] },
  ];

  it("показывает забывание ранее верного образа без сброса весов", () => {
    let state = configurePerceptron(createNetwork(), 2, 1);
    state.neurons[0]!.constantInput = 1;
    state = trainStep(state).state;
    expect(state.weights.slice(0, 2).map(row => row[2])).toEqual([0.1, 0]);
    expect(evaluateExamples(state, simple).rows[0]!.correct).toBe(true);
    state.neurons[1]!.constantInput = 1;
    state.perceptron.desired = [0];
    state = trainStep(state).state;
    expect(state.weights.slice(0, 2).map(row => row[2])).toEqual([0, -0.1]);
    expect(evaluateExamples(state, simple).rows.map(row => row.correct)).toEqual([false, true]);
  });

  it("проверяет набор на окончательных весах, а не по промежуточным успехам", () => {
    const state = configurePerceptron(createNetwork(), 2, 1);
    state.perceptron.trainingExamples = structuredClone(simple);
    const first = trainEpoch(state);
    expect(first.blockedReason).toBeUndefined();
    expect(first.report?.correct).toBe(1);
    expect(first.report?.total).toBe(2);
    expect(first.state.perceptron.trainingSteps).toBe(2);
    expect(first.changedWeights).toBe(3);
    const second = trainEpoch(first.state);
    expect(second.report?.correct).toBe(2);
    expect(second.state.weights.slice(0, 2).map(row => row[2])).toEqual([0.1, -0.1]);
    // The selected live input is still 00 with desired=1, not the last set row.
    expect(second.state.neurons.slice(0, 2).map(neuron => neuron.constantInput)).toEqual([0, 0]);
    expect(second.state.perceptron.desired).toEqual([1]);
    expect(second.state.perceptron.trained).toBe(false);
  });

  it("делает ровно одну коррекцию на образ, даже если ему нужно несколько проходов", () => {
    const state = configurePerceptron(createNetwork(), 1, 1);
    state.neurons[1]!.threshold = 0.25;
    state.perceptron.trainingExamples = [{ name: "A", inputs: [1], desired: [1] }];
    const first = trainEpoch(state);
    expect(first.blockedReason).toBeUndefined();
    expect(first.state.weights[0]![1]).toBe(0.1);
    expect(first.state.perceptron.trainingSteps).toBe(1);
    expect(first.report!.correct).toBe(0);
    const third = trainEpoch(trainEpoch(first.state).state);
    expect(third.state.perceptron.trainingSteps).toBe(3);
    expect(third.report!.correct).toBe(1);
  });

  it("подтверждает потерю правильного ответа цифры 3 после коррекции для 4", () => {
    let state = configurePerceptron(createNetwork(), 14, 1);
    state.perceptron.trainableInputs = Array.from({ length: 12 }, (_, index) => index);
    for (let source = 0; source < 12; source += 1) {
      state.weights[source]![14] = source % 2 === 0 ? 0.05 : -0.05;
    }
    const examples = digits.map(sample => ({
      name: String(sample.digit), inputs: [...sample.bits].map(Number).concat([0, 0]), desired: [sample.target],
    }));
    for (const example of examples.slice(0, 3)) state = trainEpoch(state, [example]).state;
    const before = evaluateExamples(state, [examples[2]!]).rows[0]!;
    expect(before.potentials[0]).toBeCloseTo(0.35, 12);
    expect(before.correct).toBe(true);
    const weightsBefore = structuredClone(state.weights);
    state = trainEpoch(state, [examples[3]!]).state;
    const after = evaluateExamples(state, [examples[2]!]).rows[0]!;
    expect(after.potentials[0]).toBeCloseTo(-0.05, 12);
    expect(after.correct).toBe(false);
    for (let source = 0; source < 15; source += 1) {
      for (let target = 0; target < 15; target += 1) {
        const correction = source < 12 && target === 14 ? -0.1 * examples[3]!.inputs[source]! : 0;
        expect(state.weights[source]![target]).toBe(weightsBefore[source]![target]! + correction);
      }
    }
  });

  it("проверка не меняет веса, а устаревшая пустая маска не блокирует проход набора", () => {
    let state = configurePerceptron(createNetwork(), 2, 1);
    state.weights[0]![2] = 0.1;
    state.weights[1]![2] = -0.2;
    state.perceptron.trainableInputs = [];
    state.perceptron.trainingExamples = structuredClone(simple);
    state = runStep(state).state;
    const before = structuredClone(state);
    const report = evaluateExamples(state);
    expect(report).toEqual({
      rows: [
        { outputs: [1], potentials: [0.1], correct: true },
        { outputs: [0], potentials: [-0.1], correct: true },
      ], correct: 2, total: 2,
    });
    expect(state).toEqual(before);
    report.rows[0]!.outputs[0] = 999;
    expect(state).toEqual(before);
    const epoch = trainEpoch(state);
    expect(epoch.blockedReason).toBeUndefined();
    expect(epoch.report!.correct).toBe(2);
    expect(epoch.state.weights).toEqual(before.weights);
    expect(epoch.state.perceptron.trainableInputs).toEqual([0, 1]);
  });

  it("сохраняет выбранные входы, цели, часы и историю, обновляя текущие P/z", () => {
    let state = configurePerceptron(createNetwork(), 2, 1);
    state.neurons[0]!.constantInput = 0.25;
    state.neurons[1]!.constantInput = 0.75;
    state.perceptron.desired = [0];
    state.perceptron.trainingExamples = structuredClone(simple);
    state = runStep(runStep(state).state).state;
    const before = structuredClone(state);
    const result = trainEpoch(state);
    expect(result.blockedReason).toBeUndefined();
    expect(result.state.step).toBe(before.step);
    expect(result.state.time).toBe(before.time);
    expect(result.state.history).toEqual(before.history);
    expect(result.state.neurons.slice(0, 2).map(neuron => neuron.constantInput)).toEqual([0.25, 0.75]);
    expect(result.state.perceptron.desired).toEqual([0]);
    expect(result.state.neurons[2]!.potential).toBeCloseTo(-0.075, 12);
    expect(result.state.neurons[2]!.output).toBe(0);
    expect(result.state.perceptron.trained).toBe(true);
    expect(result.state.perceptron.trainingExamples).toEqual(simple);
    expect(state).toEqual(before);
  });

  it("явный набор десяти цифр выучивается полными проходами и переживает экспорт", () => {
    let state = lab2PerceptronPreset();
    state.perceptron.trainingExamples = digits.map(sample => ({
      name: `Цифра ${sample.digit}`, inputs: [...sample.bits].map(Number).concat([0, 0]), desired: [sample.target],
    }));
    let correct = 0;
    for (let epoch = 0; epoch < 500 && correct < 10; epoch += 1) {
      const result = trainEpoch(state);
      expect(result.blockedReason).toBeUndefined();
      state = result.state;
      correct = result.report!.correct;
    }
    expect(correct).toBe(10);
    const loaded = deserializeState(JSON.parse(serializeState(state))).state;
    expect(loaded.perceptron.trainingExamples).toEqual(state.perceptron.trainingExamples);
    expect(evaluateExamples(loaded).correct).toBe(10);
  });

  it("обучает все цели каждого сохранённого образа независимо", () => {
    let state = configurePerceptron(createNetwork(), 14, 2);
    state.perceptron.trainableInputs = Array.from({ length: 12 }, (_, index) => index);
    state.perceptron.trainingExamples = digits.map(sample => ({
      name: String(sample.digit), inputs: [...sample.bits].map(Number).concat([0, 0]),
      desired: [sample.target, sample.digit >= 8 ? 1 : 0],
    }));
    for (let source = 0; source < 12; source += 1) {
      state.weights[source]![14] = source % 2 === 0 ? 0.05 : -0.05;
      state.weights[source]![15] = -state.weights[source]![14]!;
    }
    let correct = 0;
    for (let epoch = 0; epoch < 500 && correct < 10; epoch += 1) {
      const result = trainEpoch(state);
      expect(result.blockedReason).toBeUndefined();
      state = result.state;
      correct = result.report!.correct;
    }
    expect(correct).toBe(10);
    expect(evaluateExamples(state).rows.map(row => row.outputs)).toEqual(state.perceptron.trainingExamples!.map(example => example.desired));
    expect(state.weights[12]!.slice(14)).toEqual([0, 0]);
    expect(state.weights[13]!.slice(14)).toEqual([0, 0]);
  });

  it("отклоняет пустой, противоречивый и гармонический набор без изменения проекта", () => {
    const state = configurePerceptron(createNetwork(), 2, 1);
    expect(trainingExamplesIssue(state)).toContain("хотя бы один");
    expect(() => evaluateExamples(state)).toThrowError(/хотя бы один/);
    const contradictory = [...simple, { name: "Не A", inputs: [1, 0], desired: [0] }];
    expect(trainingExamplesIssue(state, contradictory)).toContain("противоречив");
    const before = structuredClone(state);
    expect(trainEpoch(state, contradictory)).toMatchObject({ state: before, report: null, changedWeights: 0 });
    state.neurons[0]!.inputKind = "harmonic";
    expect(trainingExamplesIssue(state, simple)).toContain("постоянное");
    expect(() => evaluateExamples(state, simple)).toThrowError(/постоянное/);
    expect(state.weights).toEqual(before.weights);
  });

  it("нулевой неразрешимый образ отменяет также предыдущие поправки эпохи", () => {
    const state = configurePerceptron(createNetwork(), 2, 1);
    const result = trainEpoch(state, [simple[0]!, { name: "Ноль", inputs: [0, 0], desired: [1] }]);
    expect(result.blockedReason).toContain("Образ 2");
    expect(result.blockedReason).toContain("входные нейроны выдают 0");
    expect(result.state).toEqual(state);
    expect(result.report).toBeNull();
    expect(result.changedWeights).toBe(0);
  });

  it("переполнение второго образа откатывает всю эпоху", () => {
    const state = configurePerceptron(createNetwork(), 2, 1);
    state.perceptron.learningRate = Number.MAX_VALUE / 2;
    const result = trainEpoch(state, [
      { name: "Конечный", inputs: [0.5, 0], desired: [1] },
      { name: "Переполнение", inputs: [0, 2], desired: [1] },
    ]);
    expect(result.blockedReason).toContain("Слишком большие значения");
    expect(result.state).toEqual(state);
    expect(result.report).toBeNull();
    expect(result.changedWeights).toBe(0);
  });

  it("переполнение восстановления текущего образа тоже откатывает эпоху", () => {
    const state = configurePerceptron(createNetwork(), 1, 1);
    state.neurons[0]!.constantInput = Number.MAX_VALUE;
    state.perceptron.learningRate = 10;
    const result = trainEpoch(state, [{ name: "A", inputs: [1], desired: [1] }]);
    expect(result.blockedReason).toContain("Слишком большие значения");
    expect(result.state).toEqual(state);
  });

  it("XOR не объявляется выученным после 500 эпох", () => {
    let state = configurePerceptron(createNetwork(), 2, 1);
    const xor: TrainingExample[] = [
      { name: "00", inputs: [0, 0], desired: [0] },
      { name: "01", inputs: [0, 1], desired: [1] },
      { name: "10", inputs: [1, 0], desired: [1] },
      { name: "11", inputs: [1, 1], desired: [0] },
    ];
    for (let epoch = 0; epoch < 500; epoch += 1) {
      const result = trainEpoch(state, xor);
      expect(result.blockedReason).toBeUndefined();
      expect(result.report!.correct).toBeLessThan(4);
      state = result.state;
    }
    expect(state.perceptron.trainingSteps).toBe(2000);
    expect(evaluateExamples(state, xor).correct).toBeLessThan(4);
  });

  it("сохраняет набор из 25 входов и раскладку в 5 колонок", () => {
    const state = configurePerceptron(createNetwork(), 25, 2);
    state.perceptron.patternColumns = 5;
    state.perceptron.trainingExamples = [{
      name: "Сетка 5×5", inputs: Array.from({ length: 25 }, (_, index) => index % 2), desired: [1, 0],
    }];
    expect(validateState(state)).toBe(true);
    const loaded = deserializeState(JSON.parse(serializeState(state))).state;
    expect(loaded.perceptron.patternColumns).toBe(5);
    expect(loaded.perceptron.trainingExamples).toEqual(state.perceptron.trainingExamples);
  });

  it("старый формат 2 без новых полей продолжает импортироваться", () => {
    const state = lab2PerceptronPreset();
    expect(state.perceptron).not.toHaveProperty("trainingExamples");
    expect(state.perceptron).not.toHaveProperty("patternColumns");
    expect(validateState(state)).toBe(true);
    expect(deserializeState(JSON.parse(serializeState(state))).state).toEqual(resetRuntime(state));
  });

  it("удаляет набор и раскладку только при реальном изменении топологии", () => {
    const state = configurePerceptron(createNetwork(), 2, 1);
    state.perceptron.trainingExamples = structuredClone(simple);
    state.perceptron.patternColumns = 2;
    for (const unchanged of [resizeNetwork(state, 3), configurePerceptron(state, 2, 1), switchModel(state, "perceptron")]) {
      expect(unchanged.perceptron.trainingExamples).toEqual(simple);
      expect(unchanged.perceptron.patternColumns).toBe(2);
    }
    for (const changed of [resizeNetwork(state, 4), configurePerceptron(state, 1, 2), switchModel(state, "scatter")]) {
      expect(changed.perceptron).not.toHaveProperty("trainingExamples");
      expect(changed.perceptron).not.toHaveProperty("patternColumns");
    }
    expect(state.perceptron.trainingExamples).toEqual(simple);
  });

  it("строго валидирует размеры, числа, цели, названия и лимит списка", () => {
    const state = configurePerceptron(createNetwork(), 2, 1);
    for (const trainingExamples of [
      null, {}, [{ name: "A", inputs: [1], desired: [1] }],
      [{ name: "A", inputs: [1, Infinity], desired: [1] }],
      [{ name: "A", inputs: [1, 0], desired: [0.5] }],
      [{ name: "A", inputs: [1, 0], desired: [1, 0] }],
      [{ name: " ", inputs: [1, 0], desired: [1] }],
      Array.from({ length: MAX_TRAINING_EXAMPLES + 1 }, () => simple[0]),
    ]) {
      const candidate = { ...state, perceptron: { ...state.perceptron, trainingExamples } };
      expect(validateState(candidate)).toBe(false);
      expect(() => deserializeState(candidate)).toThrowError(/Структура файла/);
      const invalid = trainingExamples as TrainingExample[];
      expect(() => evaluateExamples(state, invalid)).toThrowError();
      expect(trainEpoch(state, invalid).state).toEqual(state);
    }
    for (const patternColumns of [0, 3, 1.5, "2", null, Infinity]) {
      expect(validateState({ ...state, perceptron: { ...state.perceptron, patternColumns } })).toBe(false);
    }
    state.perceptron.trainingExamples = Array.from({ length: MAX_TRAINING_EXAMPLES }, () => structuredClone(simple[0]!));
    expect(validateState(state)).toBe(true);
    expect(trainingExamplesIssue(state)).toBeNull();
  });
});

describe("границы обучения и защита чисел", () => {
  it("не сообщает об успехе, если выходы всех входных нейронов нулевые", () => {
    const state = configurePerceptron(createNetwork(), 14, 2);
    state.perceptron.desired = [1, 0];
    const result = trainStep(state);
    expect(trainingSupportIssue(state)).toBeNull();
    expect(result.learned).toBe(false);
    expect(result.error).toBeCloseTo(Math.SQRT1_2, 12);
    expect(result.changedWeights).toBe(0);
    expect(result.blockedReason).toContain("входные нейроны выдают 0");
    expect(result.state.weights).toEqual(state.weights);
  });

  it("поясняет небинарные выходы и неверные цели, но не блокирует устаревшую пустую маску", () => {
    const state = configurePerceptron(createNetwork(), 2, 2);
    state.neurons[3]!.activation = "sigmoid";
    expect(trainingSupportIssue(state)).toContain("H4");
    expect(trainStep(state).state).toEqual(state);
    state.neurons[3]!.activation = "hard-limit";
    state.perceptron.desired[1] = 0.3;
    expect(trainingSupportIssue(state)).toContain("бинарную цель");
    state.perceptron.desired[1] = 0;
    state.perceptron.trainableInputs = [];
    state.neurons[0]!.constantInput = 1;
    expect(trainingSupportIssue(state)).toBeNull();
    expect(validateState(state)).toBe(true);
    const reloaded = deserializeState(JSON.parse(serializeState(state))).state;
    expect(reloaded.perceptron.trainableInputs).toEqual([0, 1]);
    expect(trainStep(reloaded).blockedReason).toBeUndefined();
    expect(trainStep(state).state.weights).toEqual(trainStep(reloaded).state.weights);
  });

  it.each([0, -0.1, Number.NaN, Number.POSITIVE_INFINITY])("отклоняет неверную скорость обучения %s без изменения состояния", (rate) => {
    const state = configurePerceptron(createNetwork(), 1, 1);
    state.perceptron.learningRate = rate;
    const result = trainStep(state);
    expect(result.blockedReason).toContain("конечным положительным числом");
    expect(result.learned).toBe(false);
    expect(result.changedWeights).toBe(0);
    expect(result.state).toEqual(state);
  });

  it("отменяет все поправки при переполнении одного веса", () => {
    const state = configurePerceptron(createNetwork(), 2, 1);
    state.neurons[0]!.constantInput = 0.5;
    state.neurons[1]!.constantInput = 2;
    state.perceptron.learningRate = Number.MAX_VALUE;
    const snapshot = structuredClone(state);
    const result = trainStep(state);
    expect(result.learned).toBe(false);
    expect(result.changedWeights).toBe(0);
    expect(result.blockedReason).toContain("Слишком большие значения");
    expect(result.state).toEqual(snapshot);
    expect(state).toEqual(snapshot);
    expect(validateState(result.state)).toBe(true);
  });

  it("отменяет шаг при переполнении потенциала после конечной поправки", () => {
    const state = configurePerceptron(createNetwork(), 1, 1);
    state.neurons[0]!.constantInput = 2;
    state.perceptron.learningRate = Number.MAX_VALUE / 2;
    const result = trainStep(state);
    expect(result.learned).toBe(false);
    expect(result.changedWeights).toBe(0);
    expect(result.blockedReason).toContain("Слишком большие значения");
    expect(result.state).toEqual(state);
  });

  it("не маскирует потерю точности как нулевой вход", () => {
    const state = configurePerceptron(createNetwork(), 1, 1);
    state.neurons[0]!.constantInput = 1;
    state.weights[0]![1] = 1e100;
    state.perceptron.desired = [0];
    const result = trainStep(state);
    expect(result.changedWeights).toBe(0);
    expect(result.blockedReason).toContain("числовой точности");
  });
});

describe("формат 2 и точный импорт старых проектов", () => {
  it.each([{ mask: [0] }, { mask: [] }])("обновляет устаревшую маску $mask с уведомлением, сохраняя все остальные данные", ({ mask }) => {
    let state = configurePerceptron(createNetwork(), 2, 2);
    state.perceptron.trainableInputs = mask;
    state.neurons[0]!.constantInput = 0.25;
    state.neurons[1]!.constantInput = 1;
    state.neurons[2]!.threshold = 0.4;
    state.perceptron.desired = [0, 1];
    state.weights[0]![2] = 0.3;
    state.weights[1]![3] = -0.7;
    state.perceptron.patternColumns = 2;
    state.perceptron.trainingExamples = [{ name: "A", inputs: [1, 0], desired: [1, 0] }];
    state = runStep(state).state;
    const before = structuredClone(state);
    const imported = deserializeState(state);
    expect(imported.notice).toContain("При дальнейшем обучении будут изменяться связи всех входов");
    expect(imported.notice).toContain("Сохранённые веса и параметры не изменены");
    expect(imported.state).toEqual({
      ...before, perceptron: { ...before.perceptron, trainableInputs: [0, 1] },
    });
    expect(state).toEqual(before);
    expect(deserializeState(imported.state).notice).toBeUndefined();
    const serialized = JSON.parse(serializeState(before)) as NetworkState;
    expect(serialized.perceptron.trainableInputs).toEqual([0, 1]);
    expect(serialized.weights).toEqual(before.weights);
    expect(serialized.perceptron.trainingExamples).toEqual(before.perceptron.trainingExamples);
    expect(state).toEqual(before);
  });

  it("не нормализует устаревшее поле в свободной сети и не меняет её расчёт", () => {
    const state = createNetwork(3);
    state.perceptron.trainableInputs = [];
    state.neurons[0]!.role = "input";
    state.neurons[0]!.activation = "linear";
    state.neurons[0]!.constantInput = 1;
    state.weights[0]![2] = 0.5;
    const imported = deserializeState(state);
    expect(imported.notice).toBeUndefined();
    expect(imported.state).toEqual(state);
    expect(runStep(imported.state)).toEqual(runStep(state));
    expect(JSON.parse(serializeState(state)).perceptron.trainableInputs).toEqual([]);
  });

  it("принимает новый формат и отвергает скрытые роли/связи/метаданные", () => {
    const state = configurePerceptron(createNetwork(), 2, 2);
    expect(validateState(state)).toBe(true);
    expect(validateState({ ...state, schemaVersion: 1 })).toBe(false);
    expect(validateState({ ...state, perceptron: { ...state.perceptron, ordinary: 0 } })).toBe(false);
    state.weights[2]![3] = 1;
    expect(validateState(state)).toBe(false);
    state.weights[2]![3] = 0;
    state.neurons[3]!.role = "ordinary";
    expect(validateState(state)).toBe(false);
  });

  it("сохраняет всю свободную сеть версии 1, включая обратные связи", () => {
    const current = runStep(createNetwork(3)).state;
    current.weights[2]![0] = 0.7;
    const legacy = { ...current, schemaVersion: 1, perceptron: { ...current.perceptron, ordinary: 0 } };
    const snapshot = structuredClone(legacy);
    const imported = deserializeState(legacy);
    expect(imported.notice).toContain("Все нейроны");
    expect(imported.state.weights).toEqual(current.weights);
    expect(imported.state.neurons).toEqual(current.neurons);
    expect(imported.state.history).toEqual(current.history);
    expect(validateState(imported.state)).toBe(true);
    expect(legacy).toEqual(snapshot);
  });

  it("переносит обученный legacy H16 в H15 и сохраняет 10/10 без переобучения", () => {
    const legacy = legacyFixture();
    const snapshot = structuredClone(legacy);
    const { state, notice } = deserializeState(legacy);
    expect(state.schemaVersion).toBe(2);
    expect(state.neurons).toHaveLength(15);
    expect(state.neurons[14]!.role).toBe("output");
    expect(state.neurons[14]!.threshold).toBe(legacy.neurons[15]!.threshold);
    expect(state.neurons[14]!.activation).toBe(legacy.neurons[15]!.activation);
    expect(state.neurons[14]!.linear).toEqual(legacy.neurons[15]!.linear);
    expect(state.neurons.map(neuron => neuron.harmonic)).toEqual([
      ...legacy.neurons.slice(0, 14).map(neuron => neuron.harmonic), legacy.neurons[15]!.harmonic,
    ]);
    expect(state.neurons.every(neuron => neuron.harmonic.phase === 0 && neuron.harmonic.deltaTime === 1)).toBe(true);
    expect(state.perceptron.trainableInputs).toEqual(Array.from({ length: 14 }, (_, index) => index));
    expect(state.weights.slice(0, 14).map(row => row[14])).toEqual(legacy.weights.slice(0, 14).map(row => row[15]));
    expect(state.graph.selected).toEqual([14]);
    expect(state.history).toEqual([]);
    expect(state.step).toBe(0);
    expect(notice).toContain("H16");
    expect(notice).toContain("При дальнейшем обучении будут изменяться связи всех входов");
    expect(classifyDigits(state)).toEqual(digits.map(sample => [sample.target]));
    expect(classifyDigits(deserializeState(JSON.parse(serializeState(state))).state)).toEqual(digits.map(sample => [sample.target]));
    expect(legacy).toEqual(snapshot);
  });

  it("сохраняет произвольный порог старого классификатора и прогнозы", () => {
    const legacy = legacyFixture();
    legacy.neurons[15]!.threshold = 0.123;
    const { state } = deserializeState(legacy);
    expect(state.neurons[14]!.threshold).toBe(0.123);
    const expected = digits.map(({ bits }) => {
      const potential = [...bits].reduce((sum, bit, source) => sum + Number(bit) * legacy.weights[source]![15]!, 0);
      return [potential > 0.123 ? 1 : 0];
    });
    expect(classifyDigits(state)).toEqual(expected);
  });

  it("отклоняет многовыходную старую топологию без изменения оригинала", () => {
    const legacy = legacyFixture();
    legacy.perceptron.inputs = 13;
    legacy.perceptron.outputs = 2;
    legacy.perceptron.desired = [1, 0];
    legacy.neurons[13]!.role = "output";
    const snapshot = structuredClone(legacy);
    expect(() => deserializeState(legacy)).toThrowError(/нельзя точно преобразовать/);
    expect(legacy).toEqual(snapshot);
  });

  it.each(["additional", "readout", "activation"])("не теряет несовместимые legacy данные молча: %s", (kind) => {
    const legacy = legacyFixture();
    if (kind === "additional") legacy.weights[0]![14] = 0.3;
    if (kind === "readout") legacy.weights[15]![14] = 0.9;
    if (kind === "activation") legacy.neurons[15]!.activation = "linear";
    const snapshot = structuredClone(legacy);
    expect(() => deserializeState(legacy)).toThrowError();
    expect(legacy).toEqual(snapshot);
  });

  it("строго отвергает повреждённые старые и новые проекты", () => {
    const old = legacyFixture();
    old.neurons[0]!.harmonic.deltaTime = 0;
    expect(() => deserializeState(old)).toThrowError(/Структура файла/);
    const current = configurePerceptron(createNetwork(), 2, 1);
    current.perceptron.trainableInputs = [0, 0];
    expect(validateState(current)).toBe(false);
    expect(() => deserializeState(current)).toThrowError(/Структура файла/);
  });
});
