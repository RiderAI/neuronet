import type {
  ActivationKind,
  EpochTrainingResult,
  ExamplesReport,
  GraphSample,
  NetworkState,
  Neuron,
  PerceptronArchitecture,
  StepResult,
  TrainingExample,
  TrainingResult,
} from "./types";

export const MIN_NEURONS = 2;
export const MAX_NEURONS = 32;
export const DEFAULT_NEURONS = 16;
export const MAX_TRAINING_EXAMPLES = 100;
export const EPSILON = 1e-9;
const NUMERIC_OVERFLOW_MESSAGE = "Слишком большие значения: вычисление вышло за числовой диапазон. Уменьшите входы, веса, параметры функций или скорость обучения.";

export function finite(value: unknown, fallback = 0): number {
  const parsed = typeof value === "number" ? value : Number(String(value).replace(",", "."));
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function createNeuron(id: number): Neuron {
  return {
    id,
    role: "ordinary",
    threshold: 0,
    activation: "hard-limit",
    linear: { slope: 1, intercept: 0 },
    inputKind: "constant",
    constantInput: 0,
    harmonic: { amplitude: 1, omega: 1, phase: 1, deltaTime: 0.01 },
    potential: 0,
    output: 0,
  };
}

export function blankWeights(size: number): number[][] {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
}

export function defaultPerceptron(total: number): PerceptronArchitecture {
  const inputs = Math.max(1, Math.min(14, total - 1));
  const outputs = 1;
  return {
    inputs,
    outputs,
    trainableInputs: Array.from({ length: inputs }, (_, index) => index),
    learningRate: 0.1,
    desired: [1],
    trained: false,
    trainingSteps: 0,
    lastError: 0,
  };
}

export function createNetwork(count = DEFAULT_NEURONS): NetworkState {
  const size = Math.round(clamp(finite(count, DEFAULT_NEURONS), MIN_NEURONS, MAX_NEURONS));
  return {
    schemaVersion: 2,
    title: "Без имени",
    model: "scatter",
    neurons: Array.from({ length: size }, (_, index) => createNeuron(index + 1)),
    weights: blankWeights(size),
    perceptron: defaultPerceptron(size),
    step: 0,
    time: 0,
    history: [],
    graph: {
      selected: [0, 1, 2, 3].filter((index) => index < size),
      kind: "lines",
      maxSamples: 120,
    },
  };
}

export function cloneState(state: NetworkState): NetworkState {
  return structuredClone(state);
}

function resetRuntimeMutable(state: NetworkState): void {
  state.step = 0;
  state.time = 0;
  state.history = [];
  for (const neuron of state.neurons) {
    neuron.potential = 0;
    neuron.output = 0;
  }
  state.perceptron.trained = false;
  state.perceptron.trainingSteps = 0;
  state.perceptron.lastError = 0;
}

export function resetRuntime(input: NetworkState): NetworkState {
  const state = cloneState(input);
  resetRuntimeMutable(state);
  return state;
}

export function zeroInputs(input: NetworkState): NetworkState {
  const state = cloneState(input);
  for (const neuron of state.neurons) {
    if (neuron.role === "input") neuron.constantInput = 0;
  }
  resetRuntimeMutable(state);
  return state;
}

export function resizeNetwork(input: NetworkState, requestedSize: number): NetworkState {
  const size = Math.round(
    clamp(finite(requestedSize, input.neurons.length), MIN_NEURONS, MAX_NEURONS),
  );
  const state = cloneState(input);
  const previousSize = state.neurons.length;
  if (size === previousSize) return state;
  clearTrainingExamplesMutable(state);

  if (size > previousSize) {
    for (let index = previousSize; index < size; index += 1) {
      state.neurons.push(createNeuron(index + 1));
    }
  } else {
    state.neurons.length = size;
  }

  const nextWeights = blankWeights(size);
  for (let source = 0; source < Math.min(previousSize, size); source += 1) {
    for (let target = 0; target < Math.min(previousSize, size); target += 1) {
      nextWeights[source]![target] = finite(state.weights[source]?.[target]);
    }
  }
  state.weights = nextWeights;
  state.graph.selected = state.graph.selected.filter((index) => index < size);

  const inputs = Math.min(state.perceptron.inputs, size - 1);
  const outputs = state.model === "perceptron"
    ? size - inputs
    : Math.min(state.perceptron.outputs, size - inputs);
  state.perceptron.inputs = inputs;
  state.perceptron.outputs = outputs;
  state.perceptron.trainableInputs = state.model === "perceptron"
    ? Array.from({ length: inputs }, (_, index) => index)
    : state.perceptron.trainableInputs.filter(
      (index) => Number.isInteger(index) && index >= 0 && index < inputs,
    );
  if (state.perceptron.trainableInputs.length === 0) {
    state.perceptron.trainableInputs = Array.from({ length: inputs }, (_, index) => index);
  }
  state.perceptron.desired = Array.from(
    { length: outputs },
    (_, index) => finite(state.perceptron.desired[index], 1),
  );

  if (state.model === "perceptron") assignPerceptronRolesMutable(state);
  resetRuntimeMutable(state);
  return state;
}

function clearTrainingExamplesMutable(state: NetworkState): void {
  delete state.perceptron.trainingExamples;
  delete state.perceptron.patternColumns;
}

function assignPerceptronRolesMutable(state: NetworkState, resetThresholds = false): void {
  const { inputs } = state.perceptron;
  state.neurons.forEach((neuron, index) => {
    const nextRole = index < inputs ? "input" : "output";
    const roleChanged = neuron.role !== nextRole;
    neuron.role = nextRole;
    neuron.inputKind = "constant";
    neuron.activation = index < inputs ? "linear" : "hard-limit";
    if (index < inputs) neuron.linear = { slope: 1, intercept: 0 };
    if (resetThresholds || roleChanged) {
      neuron.threshold = 0;
    }
  });
}

export function configurePerceptron(
  input: NetworkState,
  requestedInputs: number,
  requestedOutputs: number,
): NetworkState {
  const inputs = Math.round(clamp(finite(requestedInputs, 1), 1, MAX_NEURONS - 1));
  const outputs = Math.round(clamp(finite(requestedOutputs, 1), 1, MAX_NEURONS - inputs));
  const resized = resizeNetwork(input, inputs + outputs);
  if (input.model !== "perceptron" || input.perceptron.inputs !== inputs ||
      input.perceptron.outputs !== outputs) clearTrainingExamplesMutable(resized);
  resized.model = "perceptron";
  resized.perceptron.inputs = inputs;
  resized.perceptron.outputs = outputs;
  resized.perceptron.trainableInputs = Array.from({ length: inputs }, (_, index) => index);
  resized.perceptron.desired = Array.from(
    { length: outputs },
    (_, index) => finite(resized.perceptron.desired[index], 1),
  );
  resized.weights = blankWeights(resized.neurons.length);
  assignPerceptronRolesMutable(resized, true);
  resized.graph.selected = Array.from({ length: outputs }, (_, index) => inputs + index);
  resetRuntimeMutable(resized);
  return resized;
}

export function switchModel(input: NetworkState, model: NetworkState["model"]): NetworkState {
  if (input.model === model) return cloneState(input);
  let state = cloneState(input);
  clearTrainingExamplesMutable(state);
  state.model = model;
  state.weights = blankWeights(state.neurons.length);
  for (const neuron of state.neurons) {
    neuron.constantInput = 0;
    neuron.role = "ordinary";
  }
  if (model === "perceptron") {
    const defaults = defaultPerceptron(state.neurons.length);
    state = configurePerceptron(
      state,
      defaults.inputs,
      defaults.outputs,
    );
    state.model = "perceptron";
  } else {
    resetRuntimeMutable(state);
  }
  return state;
}

export function applyActivation(
  kind: ActivationKind,
  potential: number,
  threshold: number,
  linear: Neuron["linear"] = { slope: 1, intercept: 0 },
): number {
  const argument = finite(potential) - finite(threshold);
  // The legacy routine at PE address 0x407AF0 applies the strict threshold
  // gate before choosing the activation, including linear and sigmoid.
  if (argument <= 0) return 0;
  if (kind === "hard-limit") return 1;
  if (kind === "sigmoid") {
    const exponent = clamp(-argument, -700, 700);
    return 1 / (1 + Math.exp(exponent));
  }
  return finite(linear.slope, 1) * argument + finite(linear.intercept);
}

function inputValue(neuron: Neuron, nextStep: number): number {
  if (neuron.inputKind === "constant") return finite(neuron.constantInput);
  const t = nextStep * Math.max(EPSILON, finite(neuron.harmonic.deltaTime, 1));
  return (
    finite(neuron.harmonic.amplitude, 1) *
    Math.cos(finite(neuron.harmonic.omega, 1) * t + finite(neuron.harmonic.phase))
  );
}

function evaluateInput(neuron: Neuron, nextStep: number): { potential: number; output: number } {
  const potential = inputValue(neuron, nextStep);
  return {
    potential,
    output: applyActivation(neuron.activation, potential, neuron.threshold, neuron.linear),
  };
}

function evaluateScatter(state: NetworkState, nextStep: number): void {
  const previousOutputs = state.neurons.map((neuron) => finite(neuron.output));
  const currentInputs = new Map<number, { potential: number; output: number }>();

  state.neurons.forEach((neuron, index) => {
    if (neuron.role === "input") currentInputs.set(index, evaluateInput(neuron, nextStep));
  });

  const calculated = state.neurons.map((neuron, target) => {
    const currentInput = currentInputs.get(target);
    if (currentInput) return currentInput;

    let potential = 0;
    for (let source = 0; source < state.neurons.length; source += 1) {
      const weight = finite(state.weights[source]?.[target]);
      if (Math.abs(weight) < EPSILON) continue;
      const sourceOutput = currentInputs.get(source)?.output ?? previousOutputs[source]!;
      potential += weight * sourceOutput;
    }
    return {
      potential,
      output: applyActivation(neuron.activation, potential, neuron.threshold, neuron.linear),
    };
  });

  state.neurons.forEach((neuron, index) => {
    neuron.potential = calculated[index]!.potential;
    neuron.output = calculated[index]!.output;
  });
}

function perceptronIndices(state: NetworkState): {
  inputs: number[];
  outputs: number[];
} {
  const inputCount = state.perceptron.inputs;
  const outputCount = state.perceptron.outputs;
  return {
    inputs: Array.from({ length: inputCount }, (_, index) => index),
    outputs: Array.from({ length: outputCount }, (_, index) => inputCount + index),
  };
}

function evaluatePerceptron(state: NetworkState, nextStep: number): void {
  const indices = perceptronIndices(state);
  for (const index of indices.inputs) {
    const neuron = state.neurons[index]!;
    const result = evaluateInput(neuron, nextStep);
    neuron.potential = result.potential;
    neuron.output = result.output;
  }

  for (const target of indices.outputs) {
    const neuron = state.neurons[target]!;
    const potential = indices.inputs.reduce(
      (sum, source) => sum + finite(state.weights[source]?.[target]) * state.neurons[source]!.output,
      0,
    );
    neuron.potential = potential;
    neuron.output = applyActivation(neuron.activation, potential, neuron.threshold, neuron.linear);
  }
}

function nextTime(state: NetworkState, nextStep: number): number {
  const variableInput = state.neurons.find(
    (neuron) => neuron.role === "input" && neuron.inputKind === "harmonic",
  );
  return variableInput
    ? nextStep * Math.max(EPSILON, finite(variableInput.harmonic.deltaTime, 1))
    : nextStep;
}

function addHistory(state: NetworkState): void {
  const sample: GraphSample = {
    step: state.step,
    time: state.time,
    potentials: state.neurons.map((neuron) => neuron.potential),
    outputs: state.neurons.map((neuron) => neuron.output),
  };
  state.history.push(sample);
  const maxSamples = Math.round(clamp(state.graph.maxSamples, 20, 500));
  if (state.history.length > maxSamples) state.history.splice(0, state.history.length - maxSamples);
}

function hasFiniteRuntime(state: NetworkState): boolean {
  return Number.isFinite(state.time) && state.neurons.every((neuron) =>
    Number.isFinite(neuron.potential) && Number.isFinite(neuron.output),
  );
}

export function runStep(input: NetworkState): StepResult {
  const state = cloneState(input);
  const nextStep = state.step + 1;
  if (state.model === "perceptron") evaluatePerceptron(state, nextStep);
  else evaluateScatter(state, nextStep);
  state.step = nextStep;
  state.time = nextTime(state, nextStep);
  if (!hasFiniteRuntime(state)) throw new RangeError(NUMERIC_OVERFLOW_MESSAGE);
  addHistory(state);
  return { state, outputs: state.neurons.map((neuron) => neuron.output) };
}

export function trainingSupportIssue(state: NetworkState): string | null {
  if (state.model !== "perceptron") return "Обучение доступно только для модели «Перцептрон».";
  if (!Number.isFinite(state.perceptron.learningRate) || state.perceptron.learningRate <= 0) {
    return "Скорость обучения η должна быть конечным положительным числом.";
  }
  const indices = perceptronIndices(state);
  if (indices.inputs.length < 1 || indices.outputs.length < 1 ||
      state.perceptron.inputs + state.perceptron.outputs !== state.neurons.length) {
    return "Для обучения нужна прямая схема «входы → выходы» без промежуточных нейронов.";
  }
  if (state.perceptron.desired.length !== indices.outputs.length ||
      state.perceptron.desired.some((value) => value !== 0 && value !== 1)) {
    return "Для каждого выхода задайте бинарную цель: 0 или 1.";
  }
  const nonBinary = indices.outputs.filter((index) => state.neurons[index]!.activation !== "hard-limit");
  if (nonBinary.length > 0) {
    return `Для бинарного обучения выберите знаковую активацию у выходов ${nonBinary.map(index => `H${index + 1}`).join(", ")}.`;
  }
  return null;
}

/**
 * Each output is an independent binary classifier over the input neurons.
 * All incoming weights use eta * (target - output) * input; thresholds stay
 * fixed. For one output this is the LR2 classifier rule, with the redundant
 * unit-weight readout removed. Additional outputs use exactly the same rule.
 */
export function trainStep(input: NetworkState): TrainingResult {
  const state = cloneState(input);
  const rejectStep = (blockedReason: string): TrainingResult => ({
    state: cloneState(input), learned: false, error: Number.NaN, changedWeights: 0, blockedReason,
  });
  const supportIssue = trainingSupportIssue(state);
  if (supportIssue) {
    return rejectStep(supportIssue);
  }

  const indices = perceptronIndices(state);
  evaluatePerceptron(state, Math.max(1, state.step + 1));
  if (!hasFiniteRuntime(state)) return rejectStep(NUMERIC_OVERFLOW_MESSAGE);
  const eta = state.perceptron.learningRate;
  let squaredError = 0;
  let changedWeights = 0;

  const targets = indices.outputs;
  const sources = indices.inputs;
  state.perceptron.trainableInputs = [...sources];
  for (let targetIndex = 0; targetIndex < targets.length; targetIndex += 1) {
    const target = targets[targetIndex]!;
    const delta = state.perceptron.desired[targetIndex]! - state.neurons[target]!.output;
    squaredError += delta * delta;
    for (const source of sources) {
      const previousWeight = finite(state.weights[source]![target]);
      const nextWeight = previousWeight + eta * delta * state.neurons[source]!.output;
      if (!Number.isFinite(nextWeight)) return rejectStep(NUMERIC_OVERFLOW_MESSAGE);
      if (nextWeight !== previousWeight) changedWeights += 1;
      state.weights[source]![target] = nextWeight;
    }
  }
  evaluatePerceptron(state, Math.max(1, state.step + 1));
  if (!hasFiniteRuntime(state)) return rejectStep(NUMERIC_OVERFLOW_MESSAGE);
  const error = Math.sqrt(squaredError / Math.max(1, indices.outputs.length));
  const learned = targets.every((target, index) =>
    Math.abs(state.neurons[target]!.output - state.perceptron.desired[index]!) <= EPSILON,
  );
  state.perceptron.trainingSteps += 1;
  state.perceptron.lastError = error;
  state.perceptron.trained = learned;
  const blockedReason = !learned && changedWeights === 0
    ? sources.some((source) => state.neurons[source]!.output !== 0)
      ? "Поправка весов меньше доступной числовой точности. Увеличьте скорость обучения или уменьшите масштаб весов."
      : "Коррекция невозможна: все входные нейроны выдают 0. Проверьте входной образ, функции активации и пороги входных нейронов."
    : undefined;
  return { state, learned, error, changedWeights, ...(blockedReason ? { blockedReason } : {}) };
}

function exampleStructureIssue(examples: unknown, inputs: number, outputs: number): string | null {
  if (!Array.isArray(examples)) return "Учебный набор должен быть списком образов.";
  if (examples.length > MAX_TRAINING_EXAMPLES) {
    return `В учебном наборе может быть не более ${MAX_TRAINING_EXAMPLES} образов.`;
  }
  for (let index = 0; index < examples.length; index += 1) {
    const example = examples[index] as Partial<TrainingExample> | null;
    const label = `Образ ${index + 1}`;
    if (!example || typeof example !== "object" || Array.isArray(example) ||
        typeof example.name !== "string" || example.name.trim().length === 0 || example.name.length > 100) {
      return `${label}: задайте название длиной от 1 до 100 символов.`;
    }
    if (!Array.isArray(example.inputs) || example.inputs.length !== inputs ||
        !example.inputs.every(value => typeof value === "number" && Number.isFinite(value))) {
      return `${label}: нужны ${inputs} конечных числовых значений входов.`;
    }
    if (!Array.isArray(example.desired) || example.desired.length !== outputs ||
        !example.desired.every(value => value === 0 || value === 1)) {
      return `${label}: нужны ${outputs} бинарных целей (0 или 1).`;
    }
  }
  return null;
}

/** Dataset requirements, independent of weight correction. */
export function trainingExamplesIssue(
  state: NetworkState,
  examples: TrainingExample[] = state.perceptron.trainingExamples ?? [],
): string | null {
  if (state.model !== "perceptron") return "Учебный набор доступен только для модели «Перцептрон».";
  const p = state.perceptron;
  if (p.inputs < 1 || p.outputs < 1 || p.inputs + p.outputs !== state.neurons.length) {
    return "Для учебного набора нужна прямая схема «входы → выходы».";
  }
  const structureIssue = exampleStructureIssue(examples, p.inputs, p.outputs);
  if (structureIssue) return structureIssue;
  if (examples.length === 0) return "Добавьте хотя бы один образ в учебный набор.";
  if (state.neurons.slice(0, p.inputs).some(neuron => neuron.inputKind !== "constant")) {
    return "Учебный набор содержит статические входы. Выберите постоянное внешнее воздействие у всех входных нейронов.";
  }
  const seen = new Map<string, { index: number; desired: number[] }>();
  for (const [index, example] of examples.entries()) {
    const key = JSON.stringify(example.inputs);
    const previous = seen.get(key);
    if (previous && previous.desired.some((value, output) => value !== example.desired[output])) {
      return `Образы ${previous.index + 1} и ${index + 1} имеют одинаковые входы, но разные цели. Такой набор противоречив.`;
    }
    seen.set(key, { index, desired: example.desired });
  }
  return null;
}

function enterExampleMutable(state: NetworkState, example: TrainingExample): void {
  example.inputs.forEach((value, index) => { state.neurons[index]!.constantInput = value; });
  state.perceptron.desired = [...example.desired];
}

/** Check every example against one fixed matrix without touching the project. */
export function evaluateExamples(
  input: NetworkState,
  examples: TrainingExample[] = input.perceptron.trainingExamples ?? [],
): ExamplesReport {
  const issue = trainingExamplesIssue(input, examples);
  if (issue) throw new Error(issue);
  const state = cloneState(input);
  const rows = examples.map(example => {
    enterExampleMutable(state, example);
    evaluatePerceptron(state, Math.max(1, state.step + 1));
    if (!hasFiniteRuntime(state)) throw new RangeError(NUMERIC_OVERFLOW_MESSAGE);
    const outputs = state.neurons.slice(state.perceptron.inputs).map(neuron => neuron.output);
    const potentials = state.neurons.slice(state.perceptron.inputs).map(neuron => neuron.potential);
    const correct = outputs.every((value, index) => Math.abs(value - example.desired[index]!) <= EPSILON);
    return { outputs, potentials, correct };
  });
  return { rows, correct: rows.filter(row => row.correct).length, total: rows.length };
}

/** One ordered pass, one correction per example, with whole-epoch rollback. */
export function trainEpoch(
  input: NetworkState,
  examples: TrainingExample[] = input.perceptron.trainingExamples ?? [],
): EpochTrainingResult {
  const reject = (blockedReason: string): EpochTrainingResult => ({
    state: cloneState(input), report: null, changedWeights: 0, blockedReason,
  });
  const issue = trainingExamplesIssue(input, examples) ?? trainingSupportIssue(input);
  if (issue) return reject(issue);
  let state = cloneState(input);
  let changedWeights = 0;
  try {
    for (const [index, example] of examples.entries()) {
      enterExampleMutable(state, example);
      const result = trainStep(state);
      if (result.blockedReason) return reject(`Образ ${index + 1} «${example.name}»: ${result.blockedReason}`);
      state = result.state;
      changedWeights += result.changedWeights;
    }
    const report = evaluateExamples(state, examples);
    // The set is explicit training data, not a replacement for the live input.
    input.neurons.slice(0, input.perceptron.inputs).forEach((neuron, index) => {
      state.neurons[index]!.constantInput = neuron.constantInput;
    });
    state.perceptron.desired = [...input.perceptron.desired];
    evaluatePerceptron(state, Math.max(1, state.step + 1));
    if (!hasFiniteRuntime(state)) throw new RangeError(NUMERIC_OVERFLOW_MESSAGE);
    state.perceptron.trained = state.neurons.slice(state.perceptron.inputs).every((neuron, index) =>
      Math.abs(neuron.output - state.perceptron.desired[index]!) <= EPSILON,
    );
    return { state, report, changedWeights };
  } catch (error) {
    return reject(error instanceof Error ? error.message : "Не удалось выполнить проход учебного набора.");
  }
}

interface LegacyNetworkState extends Omit<NetworkState, "schemaVersion" | "perceptron"> {
  schemaVersion: 1;
  perceptron: PerceptronArchitecture & { ordinary: number };
}

function validateStateVersion(candidate: unknown, version: 1 | 2): boolean {
  if (!candidate || typeof candidate !== "object") return false;
  const state = candidate as Partial<Omit<NetworkState, "schemaVersion" | "perceptron">> & {
    schemaVersion?: unknown;
    perceptron?: unknown;
  };
  if (
    state.schemaVersion !== version ||
    typeof state.title !== "string" ||
    state.title.length > 200 ||
    (state.model !== "scatter" && state.model !== "perceptron") ||
    !Array.isArray(state.neurons) ||
    !Array.isArray(state.weights)
  ) return false;

  const size = state.neurons.length;
  if (size < MIN_NEURONS || size > MAX_NEURONS || state.weights.length !== size) return false;
  const isFiniteNumber = (value: unknown): value is number =>
    typeof value === "number" && Number.isFinite(value);
  const isInteger = (value: unknown): value is number => Number.isInteger(value);
  const isRecord = (value: unknown): value is Record<string, unknown> =>
    Boolean(value) && typeof value === "object" && !Array.isArray(value);

  if ("layout" in state && (!isRecord(state.layout) ||
      !isInteger(state.layout.neuronColumns) || state.layout.neuronColumns < 1 ||
      state.layout.neuronColumns > MAX_NEURONS)) return false;

  if (!state.weights.every((row) =>
    Array.isArray(row) && row.length === size && row.every(isFiniteNumber))) return false;

  if (!state.neurons.every((value, index) => {
    if (!isRecord(value)) return false;
    const linear = value.linear;
    const harmonic = value.harmonic;
    return (
      value.id === index + 1 &&
      (value.role === "input" || value.role === "ordinary" || value.role === "output") &&
      (value.activation === "hard-limit" || value.activation === "linear" || value.activation === "sigmoid") &&
      (value.inputKind === "constant" || value.inputKind === "harmonic") &&
      isFiniteNumber(value.threshold) &&
      isFiniteNumber(value.constantInput) &&
      isFiniteNumber(value.potential) &&
      isFiniteNumber(value.output) &&
      isRecord(linear) &&
      isFiniteNumber(linear.slope) &&
      isFiniteNumber(linear.intercept) &&
      isRecord(harmonic) &&
      isFiniteNumber(harmonic.amplitude) &&
      isFiniteNumber(harmonic.omega) &&
      isFiniteNumber(harmonic.phase) &&
      isFiniteNumber(harmonic.deltaTime) &&
      harmonic.deltaTime > 0
    );
  })) return false;

  const p = state.perceptron;
  if (!isRecord(p) || !isInteger(p.inputs) || p.inputs < 1 ||
      !isInteger(p.outputs) || p.outputs < 1) return false;
  const inputs = p.inputs;
  const outputs = p.outputs;
  if (("trainingExamples" in p && exampleStructureIssue(p.trainingExamples, inputs, outputs) !== null) ||
      ("patternColumns" in p && (!isInteger(p.patternColumns) || p.patternColumns < 1 || p.patternColumns > inputs))) {
    return false;
  }
  if (
    !Array.isArray(p.trainableInputs) || (version === 1 && p.trainableInputs.length < 1) ||
    !p.trainableInputs.every((index) => isInteger(index) && index >= 0 && index < inputs) ||
    new Set(p.trainableInputs).size !== p.trainableInputs.length ||
    !isFiniteNumber(p.learningRate) || p.learningRate <= 0 ||
    !Array.isArray(p.desired) || p.desired.length !== outputs ||
    !p.desired.every((value) => value === 0 || value === 1) ||
    typeof p.trained !== "boolean" ||
    !isInteger(p.trainingSteps) || p.trainingSteps < 0 ||
    !isFiniteNumber(p.lastError)
  ) return false;

  if (version === 1) {
    if (!isInteger(p.ordinary) || p.ordinary < (state.model === "perceptron" ? 1 : 0) ||
        inputs + outputs + p.ordinary !== size) return false;
  } else if ("ordinary" in p || inputs + outputs > size ||
      (state.model === "perceptron" && inputs + outputs !== size)) {
    return false;
  }

  if (state.model === "perceptron" && !state.neurons.every((neuron, index) => {
    const expectedRole = index < inputs
      ? "input"
      : index < inputs + outputs
        ? "output"
        : "ordinary";
    return neuron.role === expectedRole;
  })) return false;

  // A direct perceptron has only input-to-output connections. Reject hidden
  // or recurrent coefficients rather than silently ignoring imported data.
  if (version === 2 && state.model === "perceptron" && !state.weights.every((row, source) =>
    row.every((weight, target) => weight === 0 || (source < inputs && target >= inputs)),
  )) return false;

  if (!isInteger(state.step) || state.step < 0 || !isFiniteNumber(state.time)) return false;
  if (!Array.isArray(state.history) || state.history.length > 500) return false;
  if (!state.history.every((value) => {
    if (!isRecord(value)) return false;
    return (
      isInteger(value.step) && value.step >= 0 &&
      isFiniteNumber(value.time) &&
      Array.isArray(value.potentials) && value.potentials.length === size && value.potentials.every(isFiniteNumber) &&
      Array.isArray(value.outputs) && value.outputs.length === size && value.outputs.every(isFiniteNumber)
    );
  })) return false;

  const graph = state.graph;
  return (
    isRecord(graph) &&
    (graph.kind === "lines" || graph.kind === "bars") &&
    isInteger(graph.maxSamples) && graph.maxSamples >= 20 && graph.maxSamples <= 500 &&
    Array.isArray(graph.selected) &&
    graph.selected.every((index) => isInteger(index) && index >= 0 && index < size) &&
    new Set(graph.selected).size === graph.selected.length
  );
}

export function validateState(candidate: unknown): candidate is NetworkState {
  return validateStateVersion(candidate, 2);
}

function normalizeLearningInputsMutable(state: NetworkState): string | undefined {
  if (state.model !== "perceptron") return undefined;
  const allInputs = Array.from({ length: state.perceptron.inputs }, (_, index) => index);
  const hadExcludedInputs = allInputs.some(index => !state.perceptron.trainableInputs.includes(index));
  state.perceptron.trainableInputs = allInputs;
  return hadExcludedInputs
    ? "В проекте были исключённые из обучения входы. При дальнейшем обучении будут изменяться связи всех входов. Сохранённые веса и параметры не изменены."
    : undefined;
}

/** Load a validated project without silently changing an unsupported topology. */
export function deserializeState(candidate: unknown): { state: NetworkState; notice?: string } {
  if (validateState(candidate)) {
    const state = cloneState(candidate);
    const notice = normalizeLearningInputsMutable(state);
    return { state, ...(notice ? { notice } : {}) };
  }
  if (!validateStateVersion(candidate, 1)) {
    throw new Error("Структура файла не соответствует NeuroNet Web версии 1 или 2.");
  }

  const legacy = structuredClone(candidate) as LegacyNetworkState;
  const { ordinary, ...perceptron } = legacy.perceptron;
  const upgraded: NetworkState = { ...legacy, schemaVersion: 2, perceptron };
  if (legacy.model === "scatter") {
    return { state: upgraded, notice: "Проект «Россыпь» обновлён до формата 2. Все нейроны, параметры и связи сохранены." };
  }

  if (ordinary !== 1 || perceptron.outputs !== 1) {
    throw new Error("Этот проект версии 1 содержит промежуточный слой, который нельзя точно преобразовать в прямой перцептрон. Откройте его в прежней версии NeuroNet Web; исходный файл сохранён без изменений.");
  }
  const output = perceptron.inputs;
  const classifier = output + 1;
  const readout = legacy.neurons[output]!;
  const learnedNeuron = legacy.neurons[classifier]!;
  if (learnedNeuron.activation !== "hard-limit" || readout.activation !== "hard-limit" ||
      readout.threshold < 0 || readout.threshold >= 1 || legacy.weights[classifier]![output] !== 1) {
    throw new Error("Проект версии 1 не содержит точного бинарного повторителя классификатора. Без изменения результатов автоматически убрать его нельзя; используйте прежнюю версию приложения.");
  }
  const hasOtherConnections = legacy.weights.some((row, source) => row.some((weight, target) =>
    weight !== 0 && !(source < perceptron.inputs && target === classifier) &&
    !(source === classifier && target === output),
  ));
  if (hasOtherConnections) {
    throw new Error("Матрица проекта версии 1 содержит дополнительные ненулевые связи. Автоматическое преобразование потеряло бы или изменило их смысл, поэтому импорт отменён; исходный файл не изменён.");
  }

  upgraded.neurons = [
    ...legacy.neurons.slice(0, perceptron.inputs),
    { ...learnedNeuron, id: output + 1, role: "output" },
  ];
  upgraded.weights = blankWeights(upgraded.neurons.length);
  for (let source = 0; source < perceptron.inputs; source += 1) {
    upgraded.weights[source]![output] = legacy.weights[source]![classifier]!;
  }
  upgraded.graph.selected = [...new Set(legacy.graph.selected.map(index => index === classifier ? output : index))];
  resetRuntimeMutable(upgraded);
  const learningNotice = normalizeLearningInputsMutable(upgraded);
  if (!validateState(upgraded)) throw new Error("Не удалось проверить результат преобразования старого проекта.");
  return {
    state: upgraded,
    notice: `Проект версии 1 преобразован: веса и параметры классификатора H${classifier + 1} перенесены в выход H${output + 1}, повторитель удалён. Бинарные результаты сохранены; время и график начинают новый расчёт.${learningNotice ? ` ${learningNotice}` : ""}`,
  };
}

export function serializeState(state: NetworkState): string {
  const copy = resetRuntime(state);
  normalizeLearningInputsMutable(copy);
  copy.title = state.title;
  return JSON.stringify(copy, null, 2);
}
