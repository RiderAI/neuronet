import type { NetworkState } from "./types";

const ROW_LIMIT = 8;
const SPACING = 20;
const RADIUS = 9;

function countLabel(value: number, singular: string, few: string, many: string): string {
  const last = value % 10, lastTwo = value % 100;
  return lastTwo >= 11 && lastTwo <= 14 ? many : last === 1 ? singular : last >= 2 && last <= 4 ? few : many;
}

function number(value: number): string {
  if (Math.abs(value) >= 10000 || (value !== 0 && Math.abs(value) < 0.0001)) return value.toExponential(2).replace(".", ",");
  return Number(value.toFixed(4)).toString().replace(".", ",");
}

export function studioTopologyGeometry(inputs: number, outputs: number, availableWidth = 300) {
  const width = Math.max(180, Math.round(availableWidth));
  const columns = (count: number) => Math.ceil(count / ROW_LIMIT);
  const rows = (count: number) => Math.ceil(count / columns(count));
  const height = Math.max(80, Math.max(rows(inputs), rows(outputs)) * SPACING + 20);
  const layer = (count: number, output: boolean) => {
    const rowCount = rows(count);
    return Array.from({ length: count }, (_, i) => {
      const column = Math.floor(i / rowCount);
      const items = Math.min(rowCount, count - column * rowCount);
      return {
        x: output ? width - 16 - (columns(count) - 1 - column) * SPACING : 16 + column * SPACING,
        y: height / 2 + ((i % rowCount) - (items - 1) / 2) * SPACING,
      };
    });
  };
  return { width, height, inputs: layer(inputs, false), outputs: layer(outputs, true) };
}

export function renderStudioTopologyPlot(state: NetworkState, width = 300): string {
  const p = state.perceptron;
  const layout = studioTopologyGeometry(p.inputs, p.outputs, width);
  const paths = layout.inputs.flatMap((source, sourceIndex) => layout.outputs.map((target, outputIndex) => {
    const targetIndex = p.inputs + outputIndex;
    const weight = state.weights[sourceIndex]?.[targetIndex] ?? 0;
    const sourceOutput = state.neurons[sourceIndex]!.output;
    const start = source.x + RADIUS, end = target.x - RADIUS;
    const bend = Math.max(12, (end - start) * 0.45);
    return `<path class="studio-edge ${weight === 0 ? "zero" : "connected"}${weight !== 0 && Math.abs(sourceOutput) > 1e-9 ? " signal" : ""}${weight < 0 ? " inhibitory" : ""}"
      data-topology-source="${sourceIndex}" data-topology-target="${targetIndex}"
      d="M${start},${source.y} C${start + bend},${source.y} ${end - bend},${target.y} ${end},${target.y}"><title>H${sourceIndex + 1} → H${targetIndex + 1}: w = ${number(weight)}</title></path>`;
  })).join("");
  const node = (index: number, point: {x: number; y: number}) => {
    const neuron = state.neurons[index]!;
    const activity = neuron.output < -1e-9 ? " negative" : neuron.output > 1e-9 ? " signal" : "";
    return `<g class="studio-topology-neuron" data-topology-neuron="${index}"><title>H${index + 1}: P = ${number(neuron.potential)}, z = ${number(neuron.output)}</title><circle class="studio-node${activity}" cx="${point.x}" cy="${point.y}" r="${RADIUS}"/><text class="studio-node-label${activity}" x="${point.x}" y="${point.y}">${index + 1}</text></g>`;
  };
  const nodes = layout.inputs.map((point, index) => node(index, point)).join("")
    + layout.outputs.map((point, index) => node(p.inputs + index, point)).join("");
  return `<svg class="studio-topology-svg" viewBox="0 0 ${layout.width} ${layout.height}" style="--studio-topology-height:${layout.height}px" role="img" aria-label="Связи входов и выходов. Кружки — нейроны; числа — их номера."><g>${paths}</g>${nodes}</svg>`;
}

export function renderStudioTopology(state: NetworkState): string {
  const { inputs, outputs } = state.perceptron;
  const range = (first: number, count: number) => count === 1 ? `H${first}` : `H${first}–H${first + count - 1}`;
  return `<div class="studio-topology" aria-label="Архитектура перцептрона"><div class="studio-topology-head"><span><strong>${inputs}</strong> ${countLabel(inputs, "вход", "входа", "входов")} <small>${range(1, inputs)}</small></span><span><strong>${outputs}</strong> ${countLabel(outputs, "выход", "выхода", "выходов")} <small>${range(inputs + 1, outputs)}</small></span></div><div class="studio-topology-plot">${renderStudioTopologyPlot(state)}</div></div>`;
}

const observed = new Set<HTMLElement>();
let currentRoot: HTMLElement | undefined;
let currentState: NetworkState | undefined;
let observer: ResizeObserver | undefined;

function draw(element: HTMLElement): void {
  if (!currentState || currentState.model !== "perceptron" || !currentRoot?.contains(element)) return;
  const width = element.getBoundingClientRect().width;
  if (width <= 0) return;
  const html = renderStudioTopologyPlot(currentState, width);
  if (element.innerHTML !== html) element.innerHTML = html;
}

/** Update only the diagram: do not replace editing controls or their focus. */
export function syncStudioTopologies(root: HTMLElement, state: NetworkState): void {
  currentRoot = root;
  currentState = state;
  observer ??= new ResizeObserver(entries => entries.forEach(entry => draw(entry.target as HTMLElement)));
  for (const element of observed) {
    if (!root.contains(element)) { observer.unobserve(element); observed.delete(element); }
  }
  root.querySelectorAll<HTMLElement>(".studio-topology-plot").forEach(element => {
    if (!observed.has(element)) { observed.add(element); observer!.observe(element); }
    draw(element);
  });
}
