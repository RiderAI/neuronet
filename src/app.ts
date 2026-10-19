import {
  MAX_NEURONS,
  MAX_TRAINING_EXAMPLES,
  MIN_NEURONS,
  configurePerceptron,
  createNetwork,
  deserializeState,
  evaluateExamples,
  finite,
  resetRuntime,
  resizeNetwork,
  runStep,
  serializeState,
  switchModel,
  trainStep,
  trainEpoch,
  trainingExamplesIssue,
  trainingSupportIssue,
  zeroInputs,
} from "./engine";
import {
  LAB1_VARIANTS,
  lab1SingleNeuronPreset,
  lab1SummingPreset,
  lab2PerceptronPreset,
} from "./presets";
import type {
  ActivationKind,
  ExamplesReport,
  GraphKind,
  InputKind,
  NetworkState,
  NeuronRole,
} from "./types";
import "./styles.css";
import { renderStudioTopology, syncStudioTopologies } from "./studio-topology";
import { renderActivationPlot } from "./activation-plot";
import {
  STUDIO_COLUMN_PITCH,
  STUDIO_SPLITTER_WIDTH,
  studioColumns,
  studioColumnsThatFit,
  studioPaneWidth,
  studioPatternColumns,
} from "./studio-layout";

type Tab = "neurons" | "weights" | "vectors" | "perceptron" | "graph" | "labs";
type RuntimeMode = "stopped" | "running" | "training";
type ClassicWindowKey = `panel:${Tab}` | `neuron:${number}`;

interface ClassicPosition {
  x: number;
  y: number;
}

interface ClassicNeuronWindowState extends ClassicPosition {
  index: number;
}

interface ClassicDragState {
  key: ClassicWindowKey;
  pointerId: number;
  windowElement: HTMLElement;
  handle: HTMLElement;
  originX: number;
  originY: number;
  pointerX: number;
  pointerY: number;
}

interface ClassicSplitterDragState {
  pointerId: number;
  handle: HTMLElement;
  originWidth: number;
  pointerX: number;
}

interface PatternResizeState {
  pointerId: number;
  handle: HTMLButtonElement;
  originColumns: number;
  pointerX: number;
}

const CLASSIC_NEURON_FIELD_DEFAULT_WIDTH = 294;
const CLASSIC_NEURON_FIELD_MIN_WIDTH = 218;
const CLASSIC_NEURON_CARD_WIDTH = 72;
const CLASSIC_GRAPH_MIN_WIDTH = 320;
const CLASSIC_SPLITTER_WIDTH = 8;

const appElement = document.querySelector<HTMLDivElement>("#app");
if (!appElement) throw new Error("Не найден контейнер приложения");
const app: HTMLDivElement = appElement;
const classicMode = document.documentElement.dataset.ui === "classic";
const studioMode = document.documentElement.dataset.skin === "studio";

let state = createNetwork();
let selectedNeuron = 0;
let activeTab: Tab = "neurons";
const classicPanels = new Map<Tab, ClassicPosition>();
const classicView = { toolbar: true, status: true, neurons: true, graph: true };
let classicNeuronWindows: ClassicNeuronWindowState[] = [];
let classicWindowStack: ClassicWindowKey[] = [];
let classicDragState: ClassicDragState | null = null;
let classicSplitterDragState: ClassicSplitterDragState | null = null;
let patternResizeState: PatternResizeState | null = null;
let classicNeuronFieldPreferredWidth = studioMode ? studioPaneWidth(4) : CLASSIC_NEURON_FIELD_DEFAULT_WIDTH;
let classicNeuronFieldWidth = classicNeuronFieldPreferredWidth;
let studioVisibleColumns = 4;
let runtimeMode: RuntimeMode = "stopped";
let runTimer: number | undefined;
let trainingTimer: number | undefined;
let trainingAttempts = 0;
let lastChangedWeights = 0;
let runtimeNeedsStep = true;
let examplesReport: ExamplesReport | null = null;
let examplesEpochs = 0;
let exampleNameDraft = "";
let trainingTarget: "current" | "examples" = "current";
let helpOpen = false;
// Session-only UI mode: never serialize it into a project or browser storage.
let teacherMode = false;
let helpReturnFocus: FocusDescriptor | null = null;
let toast = "Готово.";
let toastTone: "info" | "success" | "warning" = "info";
let inputPointerInteraction = false;
let runtimeKeyboardInteraction = false;
let runtimeKeyboardReleaseTimer: number | undefined;
let inputRenderPending = false;
let inputRenderTimer: number | undefined;
let fullscreenPending = false;

interface FocusDescriptor {
  tag: string;
  attributes: Record<string, string>;
  windowKey?: string;
}

const FOCUS_ATTRIBUTES = [
  "id", "name", "data-action", "data-index", "data-tab", "data-model",
  "data-neuron-field", "data-source", "data-target", "data-vector-input",
  "data-weight", "data-perceptron-field", "data-desired", "data-graph-neuron",
  "data-graph-kind", "data-graph-max", "data-neuron-origin",
  "data-classic-splitter", "data-weight-editor", "data-view", "data-pattern-string", "data-pattern-resize", "data-example-name",
];

function captureFocus(): FocusDescriptor | null {
  const element = document.activeElement;
  if (!(element instanceof HTMLElement) || element === document.body) return null;
  const attributes: Record<string, string> = {};
  for (const name of FOCUS_ATTRIBUTES) {
    const value = element.getAttribute(name);
    if (value !== null) attributes[name] = value;
  }
  if (Object.keys(attributes).length === 0) return null;
  return { tag: element.tagName, attributes, windowKey: element.closest<HTMLElement>("[data-classic-window-key]")?.dataset.classicWindowKey };
}

function restoreFocus(descriptor: FocusDescriptor | null): void {
  if (!descriptor) return;
  const candidates = Array.from(document.querySelectorAll<HTMLElement>(descriptor.tag));
  const matches = candidates.filter((element) =>
    Object.entries(descriptor.attributes).every(([name, value]) => element.getAttribute(name) === value)
      && (!descriptor.windowKey || element.closest<HTMLElement>("[data-classic-window-key]")?.dataset.classicWindowKey === descriptor.windowKey),
  );
  const match = matches.find((element) => element.getClientRects().length > 0) ?? matches[0];
  match?.focus({ preventScroll: true });
}

function clearPendingInputRender(): void {
  inputRenderPending = false;
  if (inputRenderTimer !== undefined) window.clearTimeout(inputRenderTimer);
  inputRenderTimer = undefined;
}

// A native input change is emitted on blur, between pointerdown and click.
// Commit its state immediately, but keep the clicked control alive until click.
function renderAfterInputChange(): void {
  if (inputPointerInteraction) {
    inputRenderPending = true;
    return;
  }
  render();
}

function finishInputPointerInteraction(): void {
  inputPointerInteraction = false;
  if (!inputRenderPending || inputRenderTimer !== undefined) return;
  inputRenderTimer = window.setTimeout(() => {
    inputRenderTimer = undefined;
    if (!inputRenderPending) return;
    const openMenus = [...app.querySelectorAll<HTMLDetailsElement>(".classic-menu")].map(menu => menu.open);
    render();
    app.querySelectorAll<HTMLDetailsElement>(".classic-menu").forEach((menu, index) => {
      menu.open = openMenus[index] ?? false;
    });
  }, 0);
}

const COLORS = [
  "#d8483e",
  "#247c72",
  "#345aa8",
  "#e18735",
  "#7956a7",
  "#1685a6",
  "#9b6b28",
  "#c14776",
];

const ROLE_LABELS: Record<NeuronRole, string> = {
  input: "Входной",
  ordinary: "Промежуточный",
  output: "Выходной",
};

const ROLE_SHORT: Record<NeuronRole, string> = {
  input: "ВХ",
  ordinary: "ПР",
  output: "ВЫХ",
};

const ACTIVATION_LABELS: Record<ActivationKind, string> = {
  "hard-limit": "Знаковая (0/1)",
  linear: "Линейная",
  sigmoid: "Сигмоидальная",
};

const INPUT_LABELS: Record<InputKind, string> = {
  constant: "Константа",
  harmonic: "Переменная: A·cos(ωt+φ)",
};

function escapeHtml(value: unknown): string {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmt(value: number, digits = 4): string {
  if (!Number.isFinite(value)) return "—";
  const normalized = Math.abs(value) < 1e-12 ? 0 : value;
  return new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: digits,
    useGrouping: false,
  }).format(normalized);
}

function inputNumber(value: number): string {
  if (!Number.isFinite(value)) return "0";
  const exact = String(value);
  const rounded = Number(value.toPrecision(15));
  const compact = String(rounded);
  // Shorten floating-point tails for display only. Keep precise values near
  // integer thresholds, and never round the network state or exported data.
  return Number.isFinite(rounded) && !Number.isInteger(rounded) && compact.length + 3 < exact.length
    ? compact : exact;
}

function option(value: string, current: string, label: string): string {
  return `<option value="${value}" ${value === current ? "selected" : ""}>${label}</option>`;
}

function setToast(message: string, tone: typeof toastTone = "info"): void {
  toast = message;
  toastTone = tone;
}

// Editing a parameter is not a new experiment: retain the simulation clock,
// graph and recurrent outputs. Only the current training verdict is stale.
function parametersChanged(invalidateExamples = true): void {
  runtimeNeedsStep = true;
  state.perceptron.trained = false;
  state.perceptron.trainingSteps = 0;
  state.perceptron.lastError = 0;
  lastChangedWeights = 0;
  if (invalidateExamples) { examplesReport = null; examplesEpochs = 0; }
}

function stopAll(): void {
  if (runTimer !== undefined) window.clearInterval(runTimer);
  if (trainingTimer !== undefined) window.clearInterval(trainingTimer);
  runTimer = undefined;
  trainingTimer = undefined;
  runtimeMode = "stopped";
}

function selectNeuron(index: number): void {
  selectedNeuron = Math.max(0, Math.min(state.neurons.length - 1, index));
}

function compatibleLab2(): boolean {
  return (
    state.model === "perceptron" &&
    state.neurons.length === 15 &&
    state.perceptron.inputs === 14 &&
    state.perceptron.outputs === 1
  );
}

type ClassicToolbarIconName =
  | "new"
  | "open"
  | "save"
  | "print"
  | "matrix"
  | "neurons"
  | "vectors"
  | "graph"
  | "run"
  | "pause"
  | "step"
  | "stop"
  | "help";

function classicToolbarIcon(name: ClassicToolbarIconName): string {
  const paths: Record<ClassicToolbarIconName, string> = {
    new: `<path d="M3.5 1.5h6.2l2.8 2.8v10.2h-9z" fill="#fff" stroke="#111"/><path d="M9.5 1.8v3h2.8" fill="none" stroke="#111"/>`,
    open: `<path d="M1.5 4.5h5l1.3 1.4h6.7v7.6h-13z" fill="#f4c542" stroke="#111"/><path d="M2.5 3h4.3l1.2 1.5" fill="none" stroke="#111"/>`,
    save: `<path d="M2 1.5h11.5v13H2z" fill="#4c6486" stroke="#111"/><path d="M4 2h6.5v4H4zM4 9h7.5v4H4z" fill="#fff" stroke="#111"/><path d="M9 2v3" stroke="#111"/>`,
    print: `<path d="M4 1.5h8v4H4z" fill="#fff" stroke="#111"/><path d="M2 5.5h12v6H2z" fill="#bdbdbd" stroke="#111"/><path d="M4 9h8v5.5H4z" fill="#fff" stroke="#111"/><circle cx="12" cy="7.5" r=".7" fill="#111"/>`,
    matrix: `<rect x="1.5" y="1.5" width="13" height="13" fill="#fff" stroke="currentColor"/><path d="M1.5 5.8h13M1.5 10.2h13M5.8 1.5v13M10.2 1.5v13" stroke="currentColor"/>`,
    neurons: `<circle cx="8" cy="8" r="3.3" fill="#f20" stroke="#900"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.5 1.5M11.5 11.5L13 13M13 3l-1.5 1.5M4.5 11.5L3 13" stroke="#080"/>`,
    vectors: `<path d="M2 5.2h9M8.5 2.7 11 5.2 8.5 7.7M14 10.8H5M7.5 8.3 5 10.8l2.5 2.5" fill="none" stroke="currentColor" stroke-width="1.3"/>`,
    graph: `<path d="M2 2v12h12" fill="none" stroke="currentColor"/><polyline points="3,12 6,8 9,10 13,4" fill="none" stroke="#0000c0" stroke-width="1.5"/>`,
    run: `<path d="M4 2.5 13 8l-9 5.5z" fill="currentColor" stroke="currentColor"/>`,
    pause: `<path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/>`,
    step: `<path d="M2 8h8M7 4.5 10.5 8 7 11.5M12.5 3v10" fill="none" stroke="currentColor" stroke-width="1.4"/>`,
    stop: `<rect x="3" y="3" width="10" height="10" fill="currentColor" stroke="currentColor"/>`,
    help: `<circle cx="8" cy="8" r="7" fill="#ffef00" stroke="#947000"/><path d="M5.5 5.7a2.6 2.6 0 1 1 3.1 2.5c-.6.2-.8.7-.8 1.5" fill="none" stroke="#111" stroke-width="1.3"/><circle cx="7.8" cy="12" r=".8" fill="#111"/>`,
  };
  return `<svg class="classic-tool-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">${paths[name]}</svg>`;
}

function fullscreenIcon(): string {
  const path = document.fullscreenElement
    ? "M1 5h4V1M11 1v4h4M15 11h-4v4M5 15v-4H1"
    : "M5 1H1v4M11 1h4v4M15 11v4h-4M5 15H1v-4";
  return `<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true" focusable="false"><path d="${path}" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

function fullscreenLabel(): string {
  return document.fullscreenElement ? "Выйти из полноэкранного режима" : "На весь экран";
}

function renderFullscreenControl(): string {
  if (!document.fullscreenEnabled || !document.documentElement.requestFullscreen) return "";
  return `<div class="classic-window-buttons"><button type="button" data-action="toggle-fullscreen"
    aria-label="${fullscreenLabel()}" title="${fullscreenLabel()}" aria-pressed="${Boolean(document.fullscreenElement)}"
    ${fullscreenPending ? "disabled" : ""}>${fullscreenIcon()}</button></div>`;
}

function syncFullscreenControl(): void {
  const button = app.querySelector<HTMLButtonElement>('[data-action="toggle-fullscreen"]');
  if (!button) return;
  button.setAttribute("aria-pressed", String(Boolean(document.fullscreenElement)));
  button.setAttribute("aria-label", fullscreenLabel());
  button.title = fullscreenLabel();
  button.disabled = fullscreenPending;
  button.innerHTML = fullscreenIcon();
}

async function toggleFullscreen(): Promise<void> {
  if (fullscreenPending) return;
  fullscreenPending = true;
  syncFullscreenControl();
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch {
    setToast("Браузер не разрешил переключить полноэкранный режим. Можно воспользоваться его командой «На весь экран».", "warning");
    const status = app.querySelector<HTMLElement>(".toast");
    if (status) {
      status.className = `toast ${toastTone}`;
      status.textContent = toast;
    }
  } finally {
    fullscreenPending = false;
    syncFullscreenControl();
  }
}

function renderClassicHeader(): string {
  const running = runtimeMode !== "stopped";
  const modelCheck = (model: NetworkState["model"]) => state.model === model ? "✓" : "";
  const viewItem = (key: keyof typeof classicView, label: string) => `<button data-action="classic-toggle-view" data-view="${key}" aria-pressed="${classicView[key]}"><span>${classicView[key] ? "✓" : ""}</span>${label}</button>`;
  const panelItem = (tab: Tab, label: string) => `<button data-action="classic-toggle-panel" data-tab="${tab}" aria-pressed="${classicPanels.has(tab)}" ${tab === "perceptron" && state.model !== "perceptron" ? 'disabled title="Выберите Модели → Перцептрон"' : ""}><span>${classicPanels.has(tab) ? "✓" : ""}</span>${label}</button>`;
  return `
    <header class="classic-shell-header">
      <div class="classic-titlebar">
        <span class="classic-app-mark" aria-hidden="true"><i></i><b></b><em></em></span>
        <strong>NeuroNet Online</strong>
        ${renderFullscreenControl()}
      </div>
      <nav class="classic-menubar" aria-label="Главное меню">
        <details class="classic-menu" name="classic-main-menu"><summary>Файл</summary><div class="classic-menu-popup">
          <button data-action="new-project"><span>${classicToolbarIcon("new")}</span>Новый</button>
          <button data-action="import"><span>${classicToolbarIcon("open")}</span>Открыть…</button>
          <button data-action="export"><span>${classicToolbarIcon("save")}</span>Сохранить</button>
          <hr />
          <button data-action="print"><span>${classicToolbarIcon("print")}</span>Печать…</button>
        </div></details>
        <details class="classic-menu" name="classic-main-menu"><summary>Запуск</summary><div class="classic-menu-popup">
          <button data-action="run" ${runtimeMode === "training" ? "disabled" : ""}><span>${classicToolbarIcon(runtimeMode === "running" ? "pause" : "run")}</span>${runtimeMode === "running" ? "Пауза" : "Пуск"}</button>
          <button data-action="step" ${running ? "disabled" : ""}><span>${classicToolbarIcon("step")}</span>Один шаг</button>
          <button data-action="stop" ${!running ? "disabled" : ""}><span>${classicToolbarIcon("stop")}</span>Остановить</button>
          <hr />
          <button data-action="zero-inputs"><span>0</span>Обнулить входы</button>
        </div></details>
        <details class="classic-menu" name="classic-main-menu"><summary>Вид</summary><div class="classic-menu-popup">
          ${viewItem("toolbar", "Панель инструментов")}
          ${viewItem("status", "Строка статуса")}
          <hr />
          ${viewItem("neurons", "Нейронное поле")}
          ${panelItem("weights", "Матрица связей")}
          ${panelItem("neurons", "Порог и активационная функция")}
          ${panelItem("vectors", "Входные и выходные векторы")}
          ${viewItem("graph", "График")}
          ${panelItem("graph", "Свойства графика")}
          ${panelItem("perceptron", "Перцептрон и обучение")}
          ${teacherMode ? panelItem("labs", "Сценарии преподавателя") : ""}
        </div></details>
        <details class="classic-menu" name="classic-main-menu"><summary>Модели</summary><div class="classic-menu-popup">
          <button data-action="switch-model" data-model="scatter"><span>${modelCheck("scatter")}</span>Россыпь</button>
          <button data-action="switch-model" data-model="perceptron"><span>${modelCheck("perceptron")}</span>Перцептрон</button>
        </div></details>
        <details class="classic-menu" name="classic-main-menu"><summary>Справка</summary><div class="classic-menu-popup">
          <button data-action="help"><span>?</span>Справка</button>
        </div></details>
      </nav>
      <div class="classic-toolbar" role="toolbar" aria-label="Панель инструментов" ${classicView.toolbar ? "" : "hidden"}>
        <button class="classic-tool" data-action="new-project" title="Новый проект" aria-label="Новый проект">${classicToolbarIcon("new")}</button>
        <button class="classic-tool" data-action="import" title="Открыть" aria-label="Открыть">${classicToolbarIcon("open")}</button>
        <button class="classic-tool" data-action="export" title="Сохранить" aria-label="Сохранить">${classicToolbarIcon("save")}</button>
        <button class="classic-tool" data-action="print" title="Печать" aria-label="Печать">${classicToolbarIcon("print")}</button>
        <i class="classic-separator" aria-hidden="true"></i>
        <button class="classic-tool" data-action="tab" data-tab="weights" title="Матрица связей" aria-label="Матрица связей">${classicToolbarIcon("matrix")}</button>
        <button class="classic-tool" data-action="tab" data-tab="neurons" title="Порог и активация" aria-label="Порог и активация">${classicToolbarIcon("neurons")}</button>
        <button class="classic-tool" data-action="tab" data-tab="vectors" title="Входные и выходные векторы" aria-label="Входные и выходные векторы">${classicToolbarIcon("vectors")}</button>
        <button class="classic-tool" data-action="tab" data-tab="graph" title="График" aria-label="График">${classicToolbarIcon("graph")}</button>
        <button class="classic-tool" data-action="open-perceptron" title="${state.model === "perceptron" ? "Перцептрон и обучение" : "Выберите Модели → Перцептрон"}" aria-label="Перцептрон и обучение" ${state.model !== "perceptron" ? "disabled" : ""}>П</button>
        <i class="classic-separator" aria-hidden="true"></i>
        <button class="classic-tool classic-run-tool" data-action="run" title="${runtimeMode === "running" ? "Пауза" : "Пуск"}" aria-label="${runtimeMode === "running" ? "Пауза" : "Пуск"}" ${runtimeMode === "training" ? "disabled" : ""}>${classicToolbarIcon(runtimeMode === "running" ? "pause" : "run")}</button>
        <button class="classic-tool" data-action="step" title="Один шаг" aria-label="Один шаг" ${running ? "disabled" : ""}>${classicToolbarIcon("step")}</button>
        <button class="classic-tool classic-stop-tool" data-action="stop" title="Остановить" aria-label="Остановить" ${!running ? "disabled" : ""}>${classicToolbarIcon("stop")}</button>
        <button class="classic-tool classic-zero-tool" data-action="zero-inputs" title="Обнулить входы" aria-label="Обнулить входы">0</button>
        <button class="classic-tool" data-action="help" title="Справка" aria-label="Справка">${classicToolbarIcon("help")}</button>
        <i class="classic-separator" aria-hidden="true"></i>
        <div class="classic-size-control">
          <label for="network-size">Нейронов</label>
          <input id="network-size" type="number" min="${MIN_NEURONS}" max="${MAX_NEURONS}" value="${state.neurons.length}" ${state.model === "perceptron" ? "disabled" : ""} />
          <button data-action="apply-size" title="Применить число нейронов" aria-label="Применить число нейронов" ${state.model === "perceptron" || running ? "disabled" : ""}>✓</button>
        </div>
        <div class="runtime-readout" aria-label="Состояние вычисления">
          <span class="runtime-dot ${runtimeMode}"></span>
          <strong>${runtimeMode === "running" ? "Пуск" : runtimeMode === "training" ? "Обучение" : "Стоп"}</strong>
          <span>${state.step}</span>
          <span>t=${fmt(state.time)}</span>
        </div>
      </div>
      <input id="import-file" class="visually-hidden" type="file" accept="application/json,.json" />
    </header>`;
}

function renderHeader(): string {
  if (classicMode) return renderClassicHeader();
  const running = runtimeMode !== "stopped";
  return `
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true"><span></span><i></i><b></b></div>
        <div>
          <p class="eyebrow">Учебная лаборатория</p>
          <h1>NeuroNet <em>Online</em></h1>
        </div>
      </div>
      <nav class="header-actions" aria-label="Работа с проектом">
        <button class="quiet-button" data-action="new-project">Новый</button>
        <button class="quiet-button" data-action="import">Открыть</button>
        <button class="quiet-button" data-action="export">Сохранить</button>
        <button class="icon-button" data-action="print" title="Печать" aria-label="Печать">⌁</button>
        <button class="icon-button" data-action="help" title="Справка" aria-label="Справка">?</button>
      </nav>
      <input id="import-file" class="visually-hidden" type="file" accept="application/json,.json" />
    </header>
    <section class="commandbar" aria-label="Управление моделью">
      <div class="model-switch" role="group" aria-label="Модель">
        <span class="control-label">Модель</span>
        <button data-action="switch-model" data-model="scatter" aria-pressed="${state.model === "scatter"}" class="segment ${state.model === "scatter" ? "active" : ""}">Россыпь</button>
        <button data-action="switch-model" data-model="perceptron" aria-pressed="${state.model === "perceptron"}" class="segment ${state.model === "perceptron" ? "active" : ""}">Перцептрон</button>
      </div>
      <div class="run-controls" role="group" aria-label="Запуск">
        <button class="primary-button" data-action="run" ${runtimeMode === "training" ? "disabled" : ""}>
          ${runtimeMode === "running" ? "Пауза" : "Пуск"}
        </button>
        <button class="button" data-action="step" ${running ? "disabled" : ""}>Шаг</button>
        <button class="button" data-action="stop" ${!running ? "disabled" : ""}>Стоп</button>
        <button class="button" data-action="reset-runtime" ${running ? "disabled" : ""}>Сброс вычисления</button>
        <button class="button" data-action="zero-inputs">Обнулить входы</button>
      </div>
      <div class="size-control">
        <label for="network-size">Нейронов</label>
        <input id="network-size" type="number" min="${MIN_NEURONS}" max="${MAX_NEURONS}" value="${state.neurons.length}" ${state.model === "perceptron" ? "disabled" : ""} />
        <button class="button compact" data-action="apply-size" ${state.model === "perceptron" || running ? "disabled" : ""}>Изменить</button>
      </div>
      <div class="runtime-readout">
        <span class="runtime-dot ${runtimeMode}"></span>
        <strong>${runtimeMode === "running" ? "Выполняется" : runtimeMode === "training" ? "Обучение" : "Остановлено"}</strong>
        <span>шаг ${state.step}</span>
        <span>t = ${fmt(state.time)}</span>
      </div>
    </section>`;
}

function renderNeuronField(): string {
  const columns = studioMode ? Math.min(studioVisibleColumns, state.neurons.length) : Math.ceil(Math.sqrt(state.neurons.length));
  const cards = state.neurons
    .map((neuron, index) => {
      const signalClass = neuron.output > 1e-9 ? "excited" : neuron.output < -1e-9 ? "negative" : "resting";
      return `
        <button class="neuron-card ${signalClass} ${selectedNeuron === index ? "selected" : ""}"
          data-action="select-neuron" data-neuron-origin="field" data-index="${index}" aria-label="Нейрон ${neuron.id}, ${ROLE_LABELS[neuron.role]}">
          <span class="neuron-card-head">
            <strong>${classicMode && !studioMode ? neuron.id : `H${neuron.id}`}</strong>
            <small class="role role-${neuron.role}">${ROLE_SHORT[neuron.role]}</small>
          </span>
          <span class="neuron-glyph" aria-hidden="true"><i></i><b></b></span>
          <span class="neuron-values"><span>P${classicMode ? " = " : " "}<b>${fmt(neuron.potential)}</b></span><span>${classicMode && !studioMode ? "Z" : "z"}${classicMode ? " = " : " "}<b>${fmt(neuron.output)}</b></span></span>
          ${classicMode && neuron.role === "input" ? '<span class="classic-input-arrow" aria-hidden="true">&lt;--</span>' : ""}
        </button>`;
    })
    .join("");

  return `
    <article class="panel neuron-panel" id="neuron-field-panel">
      <header class="panel-heading">
        <div><p class="section-kicker">Состояние сети</p><h2>Нейронное поле</h2></div>
        <span class="legend-note"><i class="signal-swatch"></i> z &gt; 0</span>
      </header>
      <div class="neuron-grid" style="--neuron-columns:${columns};--studio-field-columns:${columns}">${cards}</div>
      <footer class="panel-foot">Нажмите нейрон, чтобы открыть его параметры. P — потенциал до порога, z — выход.</footer>
    </article>`;
}

function classicNeuronFieldBounds(): { min: number; max: number } {
  const workspaceWidth = app.querySelector<HTMLElement>(".workspace")?.getBoundingClientRect().width
    ?? window.innerWidth;
  if (studioMode) {
    const reserve = window.innerWidth <= 820 || !classicView.graph ? 0 : CLASSIC_GRAPH_MIN_WIDTH + STUDIO_SPLITTER_WIDTH;
    const minimum = studioPaneWidth(1);
    return { min: minimum, max: Math.max(minimum, Math.min(studioPaneWidth(state.neurons.length, studioFieldScrollbar()), workspaceWidth - reserve)) };
  }
  const contentMaximum = CLASSIC_NEURON_CARD_WIDTH * Math.min(MAX_NEURONS, state.neurons.length) + 2;
  const layoutMaximum = workspaceWidth - CLASSIC_GRAPH_MIN_WIDTH - CLASSIC_SPLITTER_WIDTH;
  return {
    min: CLASSIC_NEURON_FIELD_MIN_WIDTH,
    max: Math.max(CLASSIC_NEURON_FIELD_MIN_WIDTH, Math.min(contentMaximum, layoutMaximum)),
  };
}

function renderClassicWorkspaceSplitter(): string {
  const bounds = classicNeuronFieldBounds();
  return `<div class="classic-workspace-splitter" role="separator" tabindex="0" data-classic-splitter="neuron-field"
    aria-label="Ширина поля нейронов" aria-orientation="vertical" aria-controls="neuron-field-panel"
    aria-valuemin="${bounds.min}" aria-valuemax="${Math.round(bounds.max)}"
    aria-valuenow="${Math.round(classicNeuronFieldWidth)}" aria-valuetext="Ширина ${Math.round(classicNeuronFieldWidth)} пикселей"
    title="Перетащите границу, чтобы изменить число нейронов в строке"><span aria-hidden="true"></span></div>`;
}

function setClassicNeuronFieldWidth(width: number, remember = true): void {
  if (!classicMode) return;
  if (studioMode) {
    if (remember) {
      const bounds = classicNeuronFieldBounds();
      rememberStudioColumns(studioColumnsThatFit(Math.max(bounds.min, Math.min(bounds.max, width)), studioFieldScrollbar()));
    }
    syncStudioLayout();
    return;
  }
  const bounds = classicNeuronFieldBounds();
  classicNeuronFieldWidth = Math.round(Math.max(bounds.min, Math.min(bounds.max, width)));
  if (remember) classicNeuronFieldPreferredWidth = classicNeuronFieldWidth;
  const workspace = app.querySelector<HTMLElement>(".workspace");
  workspace?.style.setProperty("--classic-neuron-pane-width", `${classicNeuronFieldWidth}px`);
  const splitter = workspace?.querySelector<HTMLElement>(".classic-workspace-splitter");
  if (splitter) {
    splitter.setAttribute("aria-valuemin", String(bounds.min));
    splitter.setAttribute("aria-valuemax", String(Math.round(bounds.max)));
    splitter.setAttribute("aria-valuenow", String(classicNeuronFieldWidth));
    splitter.setAttribute("aria-valuetext", `Ширина ${classicNeuronFieldWidth} пикселей`);
  }
}

function studioPreferredColumns(): number {
  return studioColumns(state.layout?.neuronColumns ?? state.perceptron.patternColumns ?? 4);
}

function studioFieldScrollbar(): number {
  const panel = app.querySelector<HTMLElement>(".neuron-panel");
  return panel ? Math.max(0, panel.offsetWidth - panel.clientWidth) : 0;
}

function rememberStudioColumns(columns: number): void {
  const preferred = studioColumns(columns);
  state.layout = { neuronColumns: preferred };
  state.perceptron.patternColumns = studioPatternColumns(preferred, state.perceptron.inputs);
}

function prepareStudioLayout(): void {
  if (!studioMode) return;
  if (!state.layout) rememberStudioColumns(studioPreferredColumns());
  state.perceptron.patternColumns = studioPatternColumns(studioPreferredColumns(), state.perceptron.inputs);
  const maximum = classicNeuronFieldBounds().max;
  studioVisibleColumns = Math.min(studioPreferredColumns(), state.neurons.length, studioColumnsThatFit(maximum, studioFieldScrollbar()));
}

/** Layout-only updates: never replace editors, mutate signals, or clear history. */
function syncStudioLayout(): void {
  if (!studioMode || !classicMode) return;
  if (!state.layout) rememberStudioColumns(studioPreferredColumns());
  const preferred = studioPreferredColumns();
  const desired = Math.min(preferred, state.neurons.length);
  const workspace = app.querySelector<HTMLElement>(".workspace");
  const panel = workspace?.querySelector<HTMLElement>(".neuron-panel");
  const grid = panel?.querySelector<HTMLElement>(".neuron-grid");
  const bounds = classicNeuronFieldBounds();
  let visible = desired;
  // Changing rows can introduce a vertical scrollbar. Measure actual client
  // width and compensate its gutter rather than silently losing a column.
  for (let attempt = 0; attempt < 3; attempt++) {
    classicNeuronFieldPreferredWidth = studioPaneWidth(desired, studioFieldScrollbar());
    classicNeuronFieldWidth = Math.round(Math.max(bounds.min, Math.min(bounds.max, classicNeuronFieldPreferredWidth)));
    workspace?.style.setProperty("--classic-neuron-pane-width", `${classicNeuronFieldWidth}px`);
    grid?.style.setProperty("--studio-field-columns", String(visible));
    grid?.style.setProperty("--neuron-columns", String(visible));
    const available = panel && panel.clientWidth > 0 ? panel.clientWidth : classicNeuronFieldWidth - studioFieldScrollbar();
    visible = Math.min(desired, studioColumnsThatFit(available));
  }
  studioVisibleColumns = visible;
  grid?.style.setProperty("--studio-field-columns", String(visible));
  grid?.style.setProperty("--neuron-columns", String(visible));
  const splitter = workspace?.querySelector<HTMLElement>(".classic-workspace-splitter");
  splitter?.setAttribute("aria-valuemin", String(bounds.min));
  splitter?.setAttribute("aria-valuemax", String(Math.round(bounds.max)));
  splitter?.setAttribute("aria-valuenow", String(classicNeuronFieldWidth));
  splitter?.setAttribute("aria-valuetext", `${visible} столбцов нейронов`);
  updatePatternLayout(studioPatternColumns(visible, state.perceptron.inputs));
}

function classicNeuronWindowKey(index: number): ClassicWindowKey {
  return `neuron:${index}`;
}

function reconcileClassicWindowState(): void {
  if (!classicMode) return;
  if (!teacherMode) classicPanels.delete("labs");
  if (state.model !== "perceptron") classicPanels.delete("perceptron");
  classicNeuronWindows = classicNeuronWindows.filter(({ index }) => Boolean(state.neurons[index]));
  const visibleKeys: ClassicWindowKey[] = classicNeuronWindows.map(({ index }) => classicNeuronWindowKey(index));
  classicPanels.forEach((_, tab) => visibleKeys.push(`panel:${tab}`));
  const visible = new Set<ClassicWindowKey>(visibleKeys);
  classicWindowStack = classicWindowStack.filter((key) => visible.has(key));
  for (const key of visibleKeys) {
    if (!classicWindowStack.includes(key)) classicWindowStack.push(key);
  }
}

function classicWindowZ(key: ClassicWindowKey): number {
  const index = classicWindowStack.indexOf(key);
  return 1 + Math.max(0, index);
}

function activeClassicWindow(): ClassicWindowKey | undefined {
  return classicWindowStack[classicWindowStack.length - 1];
}

function syncClassicWindowLayers(): void {
  if (!classicMode) return;
  const active = activeClassicWindow();
  app.querySelectorAll<HTMLElement>("[data-classic-window-key]").forEach((element) => {
    const key = element.dataset.classicWindowKey as ClassicWindowKey;
    element.style.zIndex = String(classicWindowZ(key));
    element.dataset.classicActive = String(key === active);
  });
}

function bringClassicWindowToFront(key: ClassicWindowKey): void {
  if (!classicMode) return;
  classicWindowStack = [...classicWindowStack.filter((item) => item !== key), key];
  syncClassicWindowLayers();
}

function defaultClassicNeuronWindowPosition(order: number): ClassicPosition {
  const verticalSlots = Math.max(1, Math.min(2, Math.floor((window.innerHeight - 112) / 244)));
  const slot = order % verticalSlots;
  const stack = Math.floor(order / verticalSlots);
  const maxX = Math.max(4, window.innerWidth - 470);
  const maxY = Math.max(74, window.innerHeight - 261);
  return {
    x: Math.max(4, Math.min(maxX, (studioMode && window.innerWidth > 820 ? classicNeuronFieldWidth + STUDIO_SPLITTER_WIDTH + 12 : 312) + stack * 22)),
    y: Math.max(74, Math.min(maxY, 90 + slot * 244 + stack * 22)),
  };
}

function openClassicNeuronWindow(index: number): void {
  let windowState = classicNeuronWindows.find((item) => item.index === index);
  if (!windowState) {
    windowState = { index, ...defaultClassicNeuronWindowPosition(classicNeuronWindows.length) };
    classicNeuronWindows.push(windowState);
  }
  bringClassicWindowToFront(classicNeuronWindowKey(index));
}

function openClassicSettings(tab: Tab = activeTab): void {
  if (!classicMode) { activeTab = tab; return; }
  if (tab === "labs" && !teacherMode) return;
  if (tab === "perceptron" && state.model !== "perceptron") return;
  if (!classicPanels.has(tab)) {
    const offset = classicPanels.size * 24;
    // Leave the field's resize grip visible when the viewport has enough room.
    const x = studioMode && window.innerWidth > 820
      ? classicNeuronFieldWidth + STUDIO_SPLITTER_WIDTH + 12 + offset
      : Math.min(330 + offset, window.innerWidth * 0.22);
    classicPanels.set(tab, { x, y: 90 + offset });
  }
  bringClassicWindowToFront(`panel:${tab}`);
}

function closeClassicPanel(tab: Tab): void {
  classicPanels.delete(tab);
  removeClassicWindowFromStack(`panel:${tab}`);
}

function removeClassicWindowFromStack(key: ClassicWindowKey): void {
  classicWindowStack = classicWindowStack.filter((item) => item !== key);
}

function renderClassicActivationPlot(index: number): string {
  return renderActivationPlot(state.neurons[index]!);
}

function renderClassicNeuronWindows(): string {
  if (!classicMode) return "";
  const active = activeClassicWindow();
  const windows = classicNeuronWindows.map(({ index, x, y }) => {
    const neuron = state.neurons[index]!;
    const key = classicNeuronWindowKey(index);
    const role = ROLE_LABELS[neuron.role];
    const activation = neuron.activation === "hard-limit" ? "Знаковая" : ACTIVATION_LABELS[neuron.activation];
    const inputImpact = neuron.potential;
    return `<section class="classic-neuron-properties" role="dialog" aria-modal="false"
      aria-labelledby="classic-neuron-title-${index}" data-neuron-window-index="${index}"
      data-classic-window-key="${key}" data-classic-active="${key === active}"
      style="left:${x}px;top:${y}px;z-index:${classicWindowZ(key)}">
      <header class="classic-neuron-window-title">
        <strong id="classic-neuron-title-${index}">Свойства — H${neuron.id}</strong>
        <button data-action="classic-close-neuron" data-index="${index}" aria-label="Закрыть свойства нейрона ${neuron.id}">×</button>
      </header>
      <div class="classic-neuron-window-body">
        <section class="classic-neuron-function">
          <strong>Вид функции</strong>
          <div class="classic-neuron-function-plot">${renderClassicActivationPlot(index)}</div>
        </section>
        <section class="classic-neuron-readout">
          <button class="classic-neuron-close" data-action="classic-close-neuron" data-index="${index}">Закрыть</button>
          <dl>
            <dt>Порядковый номер</dt><dd>${neuron.id}</dd>
            <dt>Статус</dt><dd>${role}</dd>
            <dt>Порог :</dt><dd>${fmt(neuron.threshold)}</dd>
            <dt title="Суммарный сигнал до вычитания порога: P = Σw·z">Входное воздействие P :</dt><dd data-classic-live="input-impact">${fmt(inputImpact)}</dd>
            <dt>Функция активации:</dt><dd>${activation}</dd>
            <dt>Выходное значение :</dt><dd data-classic-live="output">${fmt(neuron.output)}</dd>
            <dt>Время</dt><dd data-classic-live="time">${fmt(state.time)}</dd>
          </dl>
        </section>
      </div>
    </section>`;
  }).join("");
  return `<div class="classic-neuron-windows">${windows}</div>`;
}

function graphSeries(): number[] {
  return [...new Set(state.graph.selected.filter((index) => Number.isInteger(index) && index >= 0 && index < state.neurons.length))];
}

function renderLineGraph(indices: number[]): string {
  if (indices.length === 0) {
    return `<div class="empty-graph"><strong>Нейроны для графика не выбраны</strong><span>Откройте «Настроить» и отметьте нужные нейроны.</span></div>`;
  }
  if (state.history.length === 0) {
    return `<div class="empty-graph" aria-label="Нет измерений"></div>`;
  }

  const width = 480;
  const height = 260;
  const pad = { left: 66, right: 20, top: 32, bottom: 56 };
  const values = state.history.flatMap((sample) => indices.map((index) => sample.outputs[index] ?? 0));
  // Normalise before computing the range so even finite values of opposite
  // signs near Number.MAX_VALUE cannot overflow the SVG coordinates.
  const magnitude = Math.max(...values.map(value => Math.abs(value))) || 1;
  const unit = 10 ** Math.max(-308, Math.floor(Math.log10(magnitude)));
  const rawMin = Math.min(0, ...values.map(value => value / unit));
  const rawMax = Math.max(0, ...values.map(value => value / unit));
  const roughStep = (rawMax - rawMin || 1) / 4;
  const power = 10 ** Math.floor(Math.log10(roughStep));
  const tickStep = ([1, 2, 2.5, 5, 10].find(value => value >= roughStep / power) ?? 10) * power;
  const roundedMin = Math.floor(rawMin / tickStep) * tickStep;
  const roundedMax = Math.ceil((rawMax || (rawMin === 0 ? 1 : 0)) / tickStep) * tickStep;
  const min = Number.isFinite(roundedMin * unit) ? roundedMin : rawMin;
  const max = Number.isFinite(roundedMax * unit) ? roundedMax : rawMax;
  const label = (value: number) => Math.abs(value) >= 1e6 || (value !== 0 && Math.abs(value) < 0.001)
    ? value.toExponential(2).replace(".", ",")
    : fmt(value, 4);
  const x = (sampleIndex: number) =>
    pad.left + (state.history.length === 1 ? 0.5 : sampleIndex / (state.history.length - 1)) * (width - pad.left - pad.right);
  const y = (value: number) =>
    pad.top + ((max - value / unit) / (max - min)) * (height - pad.top - pad.bottom);
  const firstTick = Math.ceil(min / tickStep - 1e-10);
  const tickCount = Math.min(8, Math.floor(max / tickStep + 1e-10) - firstTick + 1);
  const grid = Array.from({ length: tickCount }, (_, row) => {
    const value = (firstTick + row) * tickStep * unit;
    const yy = y(value);
    return `<line x1="${pad.left}" y1="${yy}" x2="${width - pad.right}" y2="${yy}"/><text class="chart-tick" x="${pad.left - 8}" y="${yy + 5}" text-anchor="end">${escapeHtml(label(value))}</text>`;
  }).join("");
  const horizontalTickCount = Math.min(5, state.history.length);
  const tickIndices = Array.from({ length: horizontalTickCount }, (_, index) =>
    Math.round(index * (state.history.length - 1) / Math.max(1, horizontalTickCount - 1)),
  );
  const timeTicks = tickIndices.map(index => {
    const sample = state.history[index]!;
    return `<line class="chart-grid-vertical" x1="${x(index)}" y1="${pad.top}" x2="${x(index)}" y2="${height - pad.bottom}"/><text class="chart-tick" x="${x(index)}" y="${height - pad.bottom + 22}" text-anchor="${state.history.length > 1 && index === 0 ? "start" : state.history.length > 1 && index === state.history.length - 1 ? "end" : "middle"}"><title>Шаг ${sample.step}; t = ${escapeHtml(fmt(sample.time, 8))}</title>${sample.step}</text>`;
  }).join("");
  const paths = indices
    .map((index) => {
      const points = state.history
        .map((sample, sampleIndex) => `${sampleIndex === 0 ? "M" : "L"}${x(sampleIndex).toFixed(2)},${y(sample.outputs[index] ?? 0).toFixed(2)}`)
        .join(" ");
      const dash = ["none", "7 3", "2 3", "7 3 2 3"][Math.floor(index / COLORS.length) % 4];
      const last = state.history.at(-1)!;
      const value = last.outputs[index] ?? 0;
      const title = `H${index + 1}: z = ${fmt(value, 8)}; шаг ${last.step}; t = ${fmt(last.time, 8)}`;
      return `<path d="${points}" stroke-dasharray="${dash}" style="--series:${COLORS[index % COLORS.length]}"><title>${escapeHtml(title)}</title></path><g class="chart-points" style="--series:${COLORS[index % COLORS.length]}"><circle cx="${x(state.history.length - 1)}" cy="${y(value)}" r="${state.history.length === 1 ? 4 : 3}"><title>${escapeHtml(title)}</title></circle></g>`;
    })
    .join("");
  return `
    <svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="График выходов нейронов">
      <title>Выходы z по расчётным шагам</title>
      <g class="chart-grid">${grid}${timeTicks}</g>
      <line class="chart-axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" />
      <line class="chart-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" />
      ${min < 0 && max > 0 ? `<line class="chart-axis" x1="${pad.left}" y1="${y(0)}" x2="${width - pad.right}" y2="${y(0)}"/>` : ""}
      <g class="chart-lines">${paths}</g>
      <text class="chart-axis-title" x="${pad.left}" y="18">Выход z</text>
      <text class="chart-axis-title" x="${width - pad.right}" y="${height - 8}" text-anchor="end">Шаг</text>
    </svg>`;
}

function renderBarGraph(indices: number[]): string {
  const latest = state.history.at(-1);
  if (!latest || indices.length === 0) return renderLineGraph(indices);
  const width = 480;
  const height = 260;
  const pad = { left: 66, right: 20, top: 32, bottom: 56 };
  const values = indices.map((index) => latest.outputs[index] ?? 0);
  const magnitude = Math.max(...values.map(value => Math.abs(value))) || 1;
  const unit = 10 ** Math.max(-308, Math.floor(Math.log10(magnitude)));
  const rawMin = Math.min(0, ...values.map(value => value / unit));
  const rawMax = Math.max(0, ...values.map(value => value / unit));
  const roughStep = (rawMax - rawMin || 1) / 4;
  const power = 10 ** Math.floor(Math.log10(roughStep));
  const tickStep = ([1, 2, 2.5, 5, 10].find(value => value >= roughStep / power) ?? 10) * power;
  const roundedMin = Math.floor(rawMin / tickStep) * tickStep;
  const roundedMax = Math.ceil((rawMax || (rawMin === 0 ? 1 : 0)) / tickStep) * tickStep;
  const min = Number.isFinite(roundedMin * unit) ? roundedMin : rawMin;
  const max = Number.isFinite(roundedMax * unit) ? roundedMax : rawMax;
  const label = (value: number) => Math.abs(value) >= 1e6 || (value !== 0 && Math.abs(value) < 0.001)
    ? value.toExponential(2).replace(".", ",")
    : fmt(value, 4);
  const y = (value: number) => pad.top + ((max - value / unit) / (max - min)) * (height - pad.top - pad.bottom);
  const zero = y(0);
  const slotWidth = (width - pad.left - pad.right) / indices.length;
  const barWidth = Math.min(64, slotWidth * 0.7);
  const firstTick = Math.ceil(min / tickStep - 1e-10);
  const tickCount = Math.min(8, Math.floor(max / tickStep + 1e-10) - firstTick + 1);
  const grid = Array.from({ length: tickCount }, (_, row) => {
    const value = (firstTick + row) * tickStep * unit;
    return `<line x1="${pad.left}" y1="${y(value)}" x2="${width - pad.right}" y2="${y(value)}"/><text class="chart-tick" x="${pad.left - 8}" y="${y(value) + 5}" text-anchor="end">${escapeHtml(label(value))}</text>`;
  }).join("");
  const horizontalTickCount = Math.min(8, indices.length);
  const tickIndices = new Set(Array.from({ length: horizontalTickCount }, (_, index) =>
    Math.round(index * (indices.length - 1) / Math.max(1, horizontalTickCount - 1)),
  ));
  const bars = indices
    .map((index, itemIndex) => {
      const value = values[itemIndex]!;
      const top = Math.min(zero, y(value));
      const h = Math.abs(zero - y(value));
      const x = pad.left + itemIndex * slotWidth + (slotWidth - barWidth) / 2;
      const title = `H${index + 1}: z = ${fmt(value, 8)}; шаг ${latest.step}; t = ${fmt(latest.time, 8)}`;
      return `<g style="--series:${COLORS[index % COLORS.length]}"><title>${escapeHtml(title)}</title>${value === 0
        ? `<line class="chart-bar-zero" x1="${x}" y1="${zero}" x2="${x + barWidth}" y2="${zero}"/>`
        : `<rect x="${x}" y="${top}" width="${barWidth}" height="${h}"/>`}
        ${indices.length <= 6 ? `<text class="chart-tick" x="${x + barWidth / 2}" y="${value >= 0 ? Math.max(27, top - 7) : Math.min(height - pad.bottom - 4, top + h + 16)}" text-anchor="middle">${escapeHtml(label(value))}</text>` : ""}
        ${tickIndices.has(itemIndex) ? `<text class="bar-label chart-tick" x="${x + barWidth / 2}" y="${height - pad.bottom + 22}" text-anchor="middle">H${index + 1}</text>` : ""}</g>`;
    })
    .join("");
  return `<svg class="chart bar-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Диаграмма выходов последнего записанного шага"><g class="chart-grid">${grid}</g><line class="chart-axis" x1="${pad.left}" y1="${zero}" x2="${width - pad.right}" y2="${zero}"/><line class="chart-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"/>${bars}<text class="chart-axis-title" x="${pad.left}" y="18">Выход z</text><text class="chart-axis-title" x="${width - pad.right}" y="${height - 8}" text-anchor="end">Нейроны</text></svg>`;
}

function renderGraphPanel(): string {
  const indices = graphSeries();
  const latest = state.history.at(-1);
  const legend = indices
    .map(
      (index) =>
        `<span title="H${index + 1}, последний записанный шаг${latest ? ` ${latest.step}` : " ещё не выполнен"}"><i style="--series:${COLORS[index % COLORS.length]};width:20px;flex-basis:20px;height:6px;background:none"><svg viewBox="0 0 20 6" width="20" height="6" aria-hidden="true"><line x1="0" y1="3" x2="20" y2="3" stroke="var(--series)" stroke-width="2" stroke-dasharray="${state.graph.kind === "bars" ? "none" : ["none", "7 3", "2 3", "7 3 2 3"][Math.floor(index / COLORS.length) % 4]}"/></svg></i><strong>H${index + 1}</strong>: ${latest ? fmt(latest.outputs[index] ?? 0) : "—"}</span>`,
    )
    .join("");
  return `
    <article class="panel graph-panel" aria-label="График выходов нейронов">
      <header class="panel-heading">
        <div class="graph-legend" aria-label="Легенда графика">${legend}</div>
        <button class="text-button" data-action="open-graph-settings">Настроить</button>
      </header>
      <div class="graph-stage">${state.graph.kind === "bars" ? renderBarGraph(indices) : renderLineGraph(indices)}</div>
      ${latest ? `<footer class="panel-foot">Шаг ${latest.step} · t = ${fmt(latest.time)} · Точек: ${state.history.length}</footer>` : ""}
    </article>`;
}

function renderTabs(): string {
  const tabs: Array<[Tab, string]> = [
    ["neurons", "Параметры нейронов"],
    ["weights", "Матрица связей"],
    ["vectors", "Векторы"],
    ["graph", "Настройки графика"],
  ];
  if (state.model === "perceptron") tabs.splice(3, 0, ["perceptron", "Перцептрон и обучение"]);
  if (teacherMode) tabs.push(["labs", "Сценарии преподавателя"]);
  if (activeTab === "labs" && !teacherMode) activeTab = "neurons";
  if (activeTab === "perceptron" && state.model !== "perceptron") activeTab = "neurons";
  return `<nav class="tabs" role="tablist">${tabs
    .map(
      ([id, label]) =>
        `<button role="tab" aria-selected="${activeTab === id}" class="tab ${activeTab === id ? "active" : ""}" data-action="tab" data-tab="${id}">${label}</button>`,
    )
    .join("")}</nav>`;
}

function renderNeuronSettings(): string {
  const neuron = state.neurons[selectedNeuron] ?? state.neurons[0]!;
  const rows = state.neurons
    .map(
      (item, index) => `
        <button class="neuron-row ${selectedNeuron === index ? "selected" : ""}" data-action="select-neuron" data-neuron-origin="settings" data-index="${index}">
          <strong>H${item.id}</strong><span class="role role-${item.role}">${ROLE_SHORT[item.role]}</span>
          <span>Q ${fmt(item.threshold)}</span><span>${ACTIVATION_LABELS[item.activation]}</span>
          <span data-neuron-potential>P ${fmt(item.potential)}</span><span data-neuron-output>z ${fmt(item.output)}</span>
        </button>`,
    )
    .join("");
  const linearFields = neuron.activation === "linear"
    ? `<fieldset class="subfields"><legend>Параметры линейной функции</legend>
        <label>Параметр k<input type="number" step="any" data-neuron-field="linear.slope" value="${inputNumber(neuron.linear.slope)}" /></label>
        <label>Параметр b<input type="number" step="any" data-neuron-field="linear.intercept" value="${inputNumber(neuron.linear.intercept)}" /></label>
      </fieldset>`
    : "";
  const inputFields = neuron.role === "input"
    ? `<fieldset class="subfields"><legend>Внешнее воздействие</legend>
        <label class="wide">Тип
          <select data-neuron-field="inputKind">
            ${option("constant", neuron.inputKind, INPUT_LABELS.constant)}
            ${option("harmonic", neuron.inputKind, INPUT_LABELS.harmonic)}
          </select>
        </label>
        ${
          neuron.inputKind === "constant"
            ? `<label>Значение x<input type="number" step="any" data-neuron-field="constantInput" value="${inputNumber(neuron.constantInput)}" /></label>`
            : `<label>Амплитуда A<input type="number" step="any" data-neuron-field="harmonic.amplitude" value="${inputNumber(neuron.harmonic.amplitude)}" /></label>
               <label>Частота ω<input type="number" step="any" data-neuron-field="harmonic.omega" value="${inputNumber(neuron.harmonic.omega)}" /></label>
               <label>Фаза φ, рад<input type="number" step="any" data-neuron-field="harmonic.phase" value="${inputNumber(neuron.harmonic.phase)}" /></label>
               <label>Шаг Δt<input type="number" step="any" min="0.000001" data-neuron-field="harmonic.deltaTime" value="${inputNumber(neuron.harmonic.deltaTime)}" /></label>`
        }
      </fieldset>`
    : `<div class="explain-card"><strong>${neuron.role === "ordinary" ? "Промежуточный нейрон" : "Выходной нейрон"}</strong><p>${
        neuron.role === "ordinary"
          ? "Получает сигналы только по связям, вычисляет активацию и передаёт результат дальше. Это внутренний, или ассоциативный, элемент."
          : "Получает сигналы по связям и включается в выходной вектор сети."
      }</p></div>`;

  return `
    <div class="settings-split">
      <section class="neuron-roster" aria-label="Список нейронов">${rows}</section>
      <section class="editor-card">
        <header><div><p class="section-kicker">Выбранный элемент</p><h3>Нейрон H${neuron.id}</h3></div><span class="live-value" data-editor-output>z = ${fmt(neuron.output)}</span></header>
        <div class="form-grid">
          <label>Статус
            <select data-neuron-field="role" ${state.model === "perceptron" ? "disabled" : ""}>
              ${option("input", neuron.role, "Входной")}
              ${state.model === "scatter" ? option("ordinary", neuron.role, ROLE_LABELS.ordinary) : ""}
              ${option("output", neuron.role, "Выходной")}
            </select>
          </label>
          <label>Порог Q<input type="number" step="any" data-neuron-field="threshold" value="${inputNumber(neuron.threshold)}" /></label>
          <label class="wide">Функция активации
            <select data-neuron-field="activation">
              ${option("hard-limit", neuron.activation, ACTIVATION_LABELS["hard-limit"])}
              ${option("linear", neuron.activation, ACTIVATION_LABELS.linear)}
              ${option("sigmoid", neuron.activation, ACTIVATION_LABELS.sigmoid)}
            </select>
          </label>
        </div>
        ${linearFields}${inputFields}
        <div class="formula-strip"><span>NET = P − Q</span><span>z = f(NET)</span><button class="text-button" data-action="help">Справка</button></div>
      </section>
    </div>`;
}

function perceptronConnectionAllowed(source: number, target: number): boolean {
  if (state.model !== "perceptron") return true;
  const inputEnd = state.perceptron.inputs;
  const outputEnd = inputEnd + state.perceptron.outputs;
  const sourceIsInput = source < inputEnd;
  const targetIsOutput = target >= inputEnd && target < outputEnd;
  return sourceIsInput && targetIsOutput;
}

function renderWeights(): string {
  const header = state.neurons.map((neuron) => `<th scope="col">H${neuron.id}</th>`).join("");
  const rows = state.neurons
    .map((sourceNeuron, source) => {
      const cells = state.neurons
        .map((_, target) => {
          const allowed = perceptronConnectionAllowed(source, target);
          return `<td class="${!allowed ? "blocked-cell" : source === target ? "diagonal-cell" : ""}">
            ${
              allowed
                ? `<input aria-label="Вес H${source + 1} в H${target + 1}" type="number" step="any" data-weight data-source="${source}" data-target="${target}" value="${inputNumber(state.weights[source]?.[target] ?? 0)}" />`
                : `<span aria-label="Связь не используется">—</span>`
            }
          </td>`;
        })
        .join("");
      return `<tr><th scope="row">H${sourceNeuron.id}</th>${cells}</tr>`;
    })
    .join("");
  return `
    <div class="section-toolbar">
      <div><h3>Матрица W[источник][приёмник]</h3><p>Строка — выход источника, столбец — вход приёмника. Например, строка H1 / столбец H2 задаёт w₁₂.</p></div>
      <button class="button" data-action="zero-weights">Обнулить матрицу</button>
    </div>
    <div class="matrix-scroll" tabindex="0">
      <table class="weight-matrix"><thead><tr><th class="corner">из → в</th>${header}</tr></thead><tbody>${rows}</tbody></table>
    </div>
    <p class="small-note">Строка — источник, столбец — приёмник. Диагональ задаёт связь нейрона с самим собой.</p>`;
}

function contextHelp(label: string, text: string): string {
  return `<button type="button" class="context-help" data-context-help="${escapeHtml(label)}" title="${escapeHtml(text)}" aria-label="${escapeHtml(label + ": " + text)}">?</button>`;
}

function renderVectors(): string {
  const rows = state.neurons
    .map(
      (neuron, index) => `<tr data-vector-neuron="${index}">
        <th scope="row">H${neuron.id}<span class="role role-${neuron.role}">${ROLE_SHORT[neuron.role]}</span></th>
        <td class="active-vector">
          ${
            neuron.role === "input" && neuron.inputKind === "constant"
              ? `<input aria-label="Вход H${neuron.id}" title="Постоянный вход. Enter или Tab — принять для следующего шага" type="number" step="any" data-vector-input data-index="${index}" value="${inputNumber(neuron.constantInput)}" />`
              : `<span data-vector-potential>${fmt(neuron.potential)}</span>`
          }
        </td>
        <td data-vector-output class="active-vector">${fmt(neuron.output)}</td>
      </tr>`,
    )
    .join("");
  return `<div class="section-toolbar"><div><h3>Входные и выходные векторы ${contextHelp("Вход и выход", "P — вход перед порогом, z — выход после активации. Постоянные входы задаются прямо в таблице и применяются на следующем шаге. Остальные значения показывают последний расчёт. Для связанных нейронов Pⱼ = Σᵢ(wᵢⱼ · zᵢ).")}</h3></div><button class="button" data-action="zero-inputs">Обнулить входы</button></div>
    <div class="simple-table-wrap"><table class="simple-table"><thead><tr><th>Нейрон</th><th>Вход нейрона P</th><th>Выход нейрона z</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function patternColumns(): number {
  const count = state.perceptron.inputs;
  if (studioMode) return studioPatternColumns(studioVisibleColumns, count);
  const square = Math.sqrt(count);
  return state.perceptron.patternColumns ?? (Number.isInteger(square) ? square : Math.min(3, count));
}

function setPatternColumns(columns: number): void {
  const count = state.perceptron.inputs;
  if (studioMode) {
    const maximum = Math.min(count, studioColumnsThatFit(classicNeuronFieldBounds().max, studioFieldScrollbar()));
    rememberStudioColumns(studioColumns(columns, maximum));
    syncStudioLayout();
    return;
  }
  columns = Math.max(1, Math.min(count, Math.round(columns)));
  // Layout only: retain the bit order, weights, results and training examples.
  state.perceptron.patternColumns = columns;
  updatePatternLayout(columns);
}

function updatePatternLayout(columns: number): void {
  const count = state.perceptron.inputs;
  app.querySelectorAll<HTMLElement>(".pattern-resizer").forEach(editor => {
    editor.style.setProperty("--pattern-columns", String(columns));
    const handle = editor.querySelector<HTMLElement>("[data-pattern-resize]");
    handle?.setAttribute("aria-valuenow", String(columns));
    if (studioMode) handle?.setAttribute("aria-valuemax", String(Math.min(count, studioColumnsThatFit(classicNeuronFieldBounds().max, studioFieldScrollbar()))));
    handle?.setAttribute("aria-valuetext", `${columns} столбцов, ${Math.ceil(count / columns)} строк`);
  });
  app.querySelectorAll<HTMLElement>("[data-pattern-dimensions]").forEach(label => {
    label.textContent = `${count} входов · ${columns} × ${Math.ceil(count / columns)}`;
  });
}

function cancelPatternResize(): void {
  if (!patternResizeState) return;
  const { handle, pointerId } = patternResizeState;
  patternResizeState = null;
  delete handle.dataset.dragging;
  if (handle.hasPointerCapture(pointerId)) handle.releasePointerCapture(pointerId);
}

function renderPatternEditor(): string {
  const count = state.perceptron.inputs;
  const columns = patternColumns();
  const bits = state.neurons.slice(0, count).map((neuron) => (neuron.constantInput >= 0.5 ? 1 : 0));
  const binaryConstant = state.neurons.slice(0, count).every(neuron => neuron.inputKind === "constant" && (neuron.constantInput === 0 || neuron.constantInput === 1));
  return `<section class="pattern-card">
    <header><div><p class="section-kicker" data-pattern-dimensions>${count} входов · ${columns} × ${Math.ceil(count / columns)}</p><h4>Редактор входного образа</h4></div></header>
    <label class="pattern-string-label">Образ, ${count} бит по строкам<input type="text" data-pattern-string aria-label="Образ, ${count} бит по строкам" inputmode="numeric" maxlength="${count}" value="${binaryConstant ? bits.join("") : ""}" placeholder="${binaryConstant ? "" : "Входы сейчас не являются постоянными 0/1"}" /></label>
    <div class="pattern-resizer" style="--pattern-columns:${columns}"><div class="pattern-grid" role="grid" aria-label="${count} компонентов входного образа">
      ${bits
        .map(
          (bit, index) =>
            `<button role="gridcell" aria-label="Вход H${index + 1}" aria-pressed="${bit === 1}" data-action="toggle-bit" data-index="${index}" class="pattern-bit ${bit ? "on" : ""}"><span>${bit}</span><small>H${index + 1}</small></button>`,
        )
        .join("")}
    </div><button type="button" class="pattern-resize-handle" data-pattern-resize role="slider" aria-label="Число столбцов входного образа" aria-orientation="horizontal" aria-valuemin="1" aria-valuemax="${count}" aria-valuenow="${columns}" aria-valuetext="${columns} столбцов, ${Math.ceil(count / columns)} строк" title="Потяните уголок влево или вправо, чтобы изменить раскладку. С клавиатуры — стрелки ← и →."></button></div>
    ${binaryConstant ? "" : '<footer>Щелчок по клетке задаёт постоянный вход 0 или 1.</footer>'}
  </section>`;
}

function renderTrainingExamples(): string {
  const examples = state.perceptron.trainingExamples ?? [];
  const issue = trainingExamplesIssue(state);
  const learningIssue = issue ?? trainingSupportIssue(state);
  return `<section class="editor-card training-examples-card">
    <header><h4>${studioMode ? "Сохранить набор" : "Учебный набор"} ${contextHelp("Учебный набор", teacherMode ? "«Обучить весь набор» по очереди корректирует веса для сохранённых образов и проверяет все ответы. Проходы повторяются до совпадения ответов или ограничения в 500 проходов. Кнопка «Прервать обучение» наверху сохраняет полученные веса. Проверка не меняет веса; не всякий набор можно выучить." : "Сохраните образы и выбирайте их для обучения по одному. «Проверить весь набор» показывает ответы на сохранённые образы, не меняя веса. Набор сохраняется в файле проекта вместе с параметрами сети.")}</h4></header>
    <div class="example-add-controls"><label>Название образа <input type="text" data-example-name maxlength="80" value="${escapeHtml(exampleNameDraft)}" placeholder="Например, буква А" /></label><button class="button" data-action="example-add">Добавить текущий образ</button></div>
    <div class="training-examples-scroll"><table class="training-examples-table"><thead><tr><th>Образ / входы</th><th>Ожидается</th><th>Получено</th><th>Совпало</th><th>Действия</th></tr></thead><tbody>${examples.map((example, index) => {
      const result = examplesReport?.rows[index];
      return `<tr data-example-index="${index}"><th>${escapeHtml(example.name)}<small>${example.inputs.map(value => fmt(value)).join(" ")}</small></th><td>${example.desired.join(" / ")}</td><td data-example-result>${result ? result.outputs.map(value => fmt(value)).join(" / ") : "—"}</td><td data-example-correct>${result ? result.correct ? "Да" : "Нет" : "—"}</td><td><button class="button" data-action="example-load" data-index="${index}">Выбрать</button><button class="button" data-action="example-remove" data-index="${index}">Удалить</button></td></tr>`;
    }).join("")}</tbody></table></div>
    <div class="training-controls examples-controls"><button class="button" data-action="examples-check" ${issue ? "disabled" : ""}>Проверить весь набор</button>${teacherMode ? `<button class="primary-button" data-action="examples-start" ${learningIssue ? "disabled" : ""}>${runtimeMode === "training" && trainingTarget === "examples" ? "Набор обучается…" : "Обучить весь набор"}</button>` : ""}</div>
    ${teacherMode && examples.length > 0 && learningIssue ? `<p class="training-problem" role="status">${escapeHtml(learningIssue)}</p>` : ""}
  </section>`;
}

function renderTrainingStatus(): string {
  const p = state.perceptron;
  const supportIssue = trainingSupportIssue(state);
  const evaluated = !runtimeNeedsStep && (state.step > 0 || p.trainingSteps > 0);
  const currentMatches = evaluated && Array.from({ length: p.outputs }, (_, i) => state.neurons[p.inputs + i]!.output === p.desired[i]).every(Boolean);
  return supportIssue
    ? `<span class="training-state"><i></i>Обучение недоступно</span>`
    : currentMatches
    ? `<span class="training-state success"><i></i>Текущий образ распознан</span>`
    : `<span class="training-state"><i></i>${runtimeMode === "training" ? "Идёт обучение" : evaluated ? "Цель ещё не достигнута" : "Образ не проверен"}</span>`;
}

function renderPerceptron(): string {
  const p = state.perceptron;
  const supportIssue = trainingSupportIssue(state);
  const inputEnd = p.inputs;
  const outputEnd = inputEnd + p.outputs;
  const status = renderTrainingStatus();
  const desired = Array.from({ length: p.outputs }, (_, index) => `
    <label>Цель d для H${p.inputs + index + 1}
      <input type="number" step="1" min="0" max="1" data-desired data-index="${index}" value="${inputNumber(p.desired[index] ?? 1)}" />
    </label>`).join("");
  const destinations = Array.from({ length: p.outputs }, (_, i) => p.inputs + i);
  const sources = Array.from({ length: p.inputs }, (_, i) => i);
  const weightRows = sources.map((source) => `
    <tr><th>H${source + 1}</th>${destinations.map(target => `<td><input aria-label="Вес H${source + 1} в H${target + 1}" title="${state.weights[source]?.[target] ?? 0}" type="number" step="any" data-weight data-weight-editor data-source="${source}" data-target="${target}" value="${inputNumber(state.weights[source]?.[target] ?? 0)}" /></td>`).join("")}</tr>`).join("");

  return `
    <div class="perceptron-layout">
      <section class="editor-card architecture-card">
        <header><div><p class="section-kicker">Топология</p><h3>Входы → выходы</h3></div>${compatibleLab2() ? '<span class="compat-badge">ЛР2</span>' : ""}</header>
        ${studioMode ? renderStudioTopology(state) : `<div class="architecture-visual" aria-label="Архитектура перцептрона">
          <div><strong>${p.inputs}</strong><span>входов</span><small>H1…H${p.inputs}</small></div><i>→</i>
          <div><strong>${p.outputs}</strong><span>выходов</span><small>H${inputEnd + 1}…H${outputEnd}</small></div>
        </div>`}
        <div class="form-grid architecture-form">
          <label>Входов<input id="arch-inputs" type="number" min="1" max="31" value="${p.inputs}" /></label>
          <label>Выходов<input id="arch-outputs" type="number" min="1" max="31" value="${p.outputs}" /></label>
          <button class="button" data-action="apply-architecture">Применить топологию</button>
        </div>
        <p class="small-note">Изменение топологии обнуляет веса и очищает учебный набор.</p>
        <div class="training-controls"><button class="button" data-action="tab" data-tab="vectors">Задать входы</button><button class="button" data-action="tab" data-tab="neurons">Пороги и функции</button><button class="button" data-action="tab" data-tab="weights">Вся матрица весов</button></div>
      </section>
      <section class="editor-card training-card">
        <header><div><h3>Обучение текущего образа ${contextHelp("Обучение", "«Одна коррекция» проверяет текущий образ и при ошибке один раз меняет веса: Δw = η(d − z)·x, где x — выход входного нейрона. «Обучить образ» повторяет коррекции до верного ответа. «Прервать обучение» останавливает обучение образа или всего набора, сохраняя веса. После обучения проверьте предыдущие образы.")}</h3></div>${status}</header>
        <div class="form-grid">
          <label>Скорость обучения η<input type="number" step="any" min="0.000001" title="Коэффициент η в поправке весов: Δw = η · (d − z) · x" data-perceptron-field="learningRate" value="${inputNumber(p.learningRate)}" /></label>
          ${desired}
        </div>
        <div class="training-controls">
          <button class="primary-button" data-action="training-start" title="Повторять коррекции до совпадения ответа с целью" ${runtimeMode === "running" || supportIssue ? "disabled" : ""}>${runtimeMode === "training" && trainingTarget === "current" ? "Обучается…" : "Обучить образ"}</button>
          <button class="button" data-action="training-step" title="Одна попытка коррекции весов для текущего образа" ${runtimeMode !== "stopped" || supportIssue ? "disabled" : ""}>Одна коррекция</button>
          <button class="button" data-action="stop" title="Прервать обучение текущего образа или всего набора, сохранив полученные веса" ${runtimeMode !== "training" ? "disabled" : ""}>Прервать обучение</button>
        </div>
        <dl class="metric-row"><div><dt>Коррекций</dt><dd data-training-steps>${p.trainingSteps}</dd></div><div><dt>Ошибка до коррекции</dt><dd>${fmt(p.lastError, 6)}</dd></div><div><dt>Изменено весов</dt><dd>${p.trainingSteps > 0 ? lastChangedWeights : "—"}</dd></div></dl>
        ${supportIssue ? `<div class="training-problem" role="status">${escapeHtml(supportIssue)}</div>` : ""}
      </section>
      ${renderPatternEditor()}
      <section class="weights-summary"><header><h4>Веса — начальные и текущие ${contextHelp("Редактирование весов", "Строка — вход, столбец — выход. Введите вес и подтвердите Enter или Tab. При обучении корректируются связи всех входов. Если входной нейрон выдаёт 0, поправка его весов равна 0.")}</h4></header>
        <div class="perceptron-weights-scroll"><table class="perceptron-weight-table"><thead><tr><th>Вход</th>${destinations.map(target => `<th>Вес в H${target + 1}</th>`).join("")}</tr></thead><tbody>${weightRows}</tbody></table></div>
      </section>
      ${renderTrainingExamples()}
    </div>`;
}

function renderGraphSettings(): string {
  return `<div class="graph-settings-grid">
    <section class="editor-card"><header><div><p class="section-kicker">Отображение</p><h3>Вид графика</h3></div></header>
      <div class="choice-row">
        <label class="radio-card"><input type="radio" name="graph-kind" data-graph-kind value="lines" ${state.graph.kind === "lines" ? "checked" : ""}/><span>Линии<small>История выходов во времени</small></span></label>
        <label class="radio-card"><input type="radio" name="graph-kind" data-graph-kind value="bars" ${state.graph.kind === "bars" ? "checked" : ""}/><span>Диаграмма<small>Последнее значение каждого выхода</small></span></label>
      </div>
      <label class="range-label">Хранить точек <input type="number" min="20" max="500" step="10" data-graph-max value="${state.graph.maxSamples}" /></label>
      <button class="button" data-action="clear-graph">Очистить график</button>
    </section>
    <section class="editor-card"><header><div><p class="section-kicker">Серии</p><h3>Отображаемые нейроны</h3></div><div><button class="text-button" data-action="graph-all">Все</button><button class="text-button" data-action="graph-none">Снять</button></div></header>
      <div class="check-grid">${state.neurons
        .map(
          (neuron, index) =>
            `<label><input type="checkbox" data-graph-neuron data-index="${index}" ${state.graph.selected.includes(index) ? "checked" : ""}/><span style="--series:${COLORS[index % COLORS.length]}">H${neuron.id}</span><small>${ROLE_SHORT[neuron.role]}</small></label>`,
        )
        .join("")}</div>
    </section>
  </div>`;
}

function renderLabs(): string {
  if (!teacherMode) return "";
  const variantOptions = LAB1_VARIANTS.map((_, index) => `<option value="${index + 1}">${index + 1}</option>`).join("");
  const thresholdVariantOptions = LAB1_VARIANTS.map((variant, index) =>
    `<option value="${index + 1}">${index + 1} — Q₂: ${fmt(variant.q21)} / ${fmt(variant.q22)} / ${fmt(variant.q23)}</option>`,
  ).join("");
  return `<div class="lab-grid">
      <article class="lab-card lab-red"><span class="lab-number">ЛР1 · 1–3</span><h4>Одиночный нейрон</h4><p>H1 → H2, w₁₂ = 1, x(t) = cos(t). Выберите активацию H2.</p>
        <label class="lab-variant-wide">Вариант<select id="lab-single-variant">${thresholdVariantOptions}</select></label>
        <div class="lab-actions"><button class="button" data-action="preset-single" data-activation="hard-limit">Знаковая</button><button class="button" data-action="preset-single" data-activation="linear">Линейная</button><button class="button" data-action="preset-single" data-activation="sigmoid">Сигмоида</button></div>
        <footer>Загружается Q₂₁; затем последовательно установите Q₂₂ и Q₂₃ и выполните по 7 шагов.</footer>
      </article>
      <article class="lab-card lab-green"><span class="lab-number">ЛР1 · 4</span><h4>Суммирующий нейрон</h4><p>H1, H2, H3 → H4; x₁ и x₂ постоянны, x₃(t) = cos(t).</p>
        <div class="lab-form"><label>Вариант<select id="lab-variant">${variantOptions}</select></label><label>F₄<select id="lab-sum-activation"><option value="hard-limit">Знаковая</option><option value="linear">Линейная</option><option value="sigmoid">Сигмоида</option></select></label><button class="primary-button" data-action="preset-summing">Загрузить</button></div>
        <footer>Веса и входы — из таблицы 3.5; исследуйте Q₄ = 0 / 0,5 / −0,2.</footer>
      </article>
      <article class="lab-card lab-blue"><span class="lab-number">ЛР2</span><h4>Чётные и нечётные цифры</h4><p>H1…H14 — входы, H15 — выходной классификатор. Промежуточного слоя нет.</p>
        <button class="primary-button" data-action="preset-lab2">Открыть перцептрон ЛР2</button>
        <footer>Включён записанный в методичке образ X₁ = 010110010010.</footer>
      </article>
      <article class="lab-card lab-neutral"><span class="lab-number">Своя схема</span><h4>Чистая сеть</h4><p>От 2 до 32 нейронов, произвольные роли, веса, пороги и самосвязи.</p>
        <button class="button" data-action="new-project">Создать пустую</button>
        <footer>По умолчанию — 16 нейронов свободной сети, как в старом NeuroNet.</footer>
      </article>
    </div>
    <div class="math-contract"><div><strong>Функции активации</strong><span>Знаковая и линейная: при P ≤ Q выход z = 0</span></div><code>При P &gt; Q: знаковая — 1; линейная — k·(P−Q)+b. Сигмоидальная при любом P: 1/(1+e<sup>−(P−Q)</sup>)</code><button class="text-button" data-action="help">Справка</button></div>`;
}

function renderActiveTab(tab: Tab = activeTab): string {
  if (tab === "neurons") return renderNeuronSettings();
  if (tab === "weights") return renderWeights();
  if (tab === "vectors") return renderVectors();
  if (tab === "perceptron") return renderPerceptron();
  if (tab === "graph") return renderGraphSettings();
  return renderLabs();
}

function classicDialogCaption(tab: Tab): string {
  const captions: Record<Tab, string> = {
    labs: "Лабораторные работы",
    neurons: "Порог и активационная функция",
    weights: "Матрица связей",
    vectors: "Входные и выходные векторы",
    perceptron: "Перцептрон",
    graph: "Свойства графика",
  };
  return captions[tab];
}

function renderClassicPanels(): string {
  return [...classicPanels.entries()].map(([tab, position]) => {
    const key: ClassicWindowKey = `panel:${tab}`;
    return `<section class="settings-panel" role="dialog" aria-modal="false" aria-label="${classicDialogCaption(tab)}"
      data-classic-tab="${tab}" data-classic-window-key="${key}" data-classic-active="${activeClassicWindow() === key}"
      style="left:${position.x}px;top:${position.y}px;transform:none;z-index:${classicWindowZ(key)}">
      <header class="classic-dialog-title"><strong>${classicDialogCaption(tab)}</strong><button data-action="classic-close" data-tab="${tab}" aria-label="Закрыть ${classicDialogCaption(tab)}">×</button></header>
      <div class="tab-runtime-lock"><div class="tab-content">${renderActiveTab(tab)}</div></div>
    </section>`;
  }).join("");
}

function updateRuntimeLocks(): void {
  if (runtimeMode === "stopped") return;
  app.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>(".tab-runtime-lock input, .tab-runtime-lock select, .tab-runtime-lock button").forEach((element) => {
    // Rearranging the Studio grid never edits signals or training parameters.
    if (studioMode && element.matches("[data-pattern-resize]")) return;
    if (runtimeMode === "running" && element.matches('[data-neuron-field], [data-vector-input], [data-weight], [data-pattern-string], [data-pattern-resize], [data-perceptron-field], [data-desired], [data-action="toggle-bit"]')) return;
    if (!element.matches('[data-context-help], [data-action="stop"], [data-action="zero-inputs"], [data-action="select-neuron"], [data-action="tab"]')) element.disabled = true;
  });
}

function capturePanelScroll(): Map<string, Array<[number, number]>> {
  const saved = new Map<string, Array<[number, number]>>();
  const legend = app.querySelector<HTMLElement>(".graph-legend");
  if (legend) saved.set("graph-legend", [[legend.scrollLeft, legend.scrollTop]]);
  app.querySelectorAll<HTMLElement>(".settings-panel").forEach((panel) => {
    saved.set(panel.dataset.classicTab ?? "modern", [...panel.querySelectorAll<HTMLElement>(".tab-runtime-lock, .matrix-scroll, .neuron-roster, .perceptron-weights-scroll, .pattern-grid, .training-examples-scroll, .simple-table-wrap")].map(el => [el.scrollLeft, el.scrollTop]));
  });
  return saved;
}

function restorePanelScroll(saved: Map<string, Array<[number, number]>>): void {
  const legend = app.querySelector<HTMLElement>(".graph-legend");
  const legendPosition = saved.get("graph-legend")?.[0];
  if (legend && legendPosition) {
    legend.scrollLeft = legendPosition[0];
    legend.scrollTop = legendPosition[1];
  }
  app.querySelectorAll<HTMLElement>(".settings-panel").forEach((panel) => {
    const values = saved.get(panel.dataset.classicTab ?? "modern");
    panel.querySelectorAll<HTMLElement>(".tab-runtime-lock, .matrix-scroll, .neuron-roster, .perceptron-weights-scroll, .pattern-grid, .training-examples-scroll, .simple-table-wrap").forEach((el, index) => {
      const position = values?.[index];
      if (position) { el.scrollLeft = position[0]; el.scrollTop = position[1]; }
    });
  });
}

function renderHelp(): string {
  if (!helpOpen) return "";
  return `<div class="modal-backdrop" data-action="close-help">
    <section class="help-modal" role="dialog" aria-modal="true" aria-labelledby="help-title" data-modal tabindex="-1">
      <header><h2 id="help-title">Справка NeuroNet Online</h2><button class="icon-button dark" data-action="close-help" aria-label="Закрыть">×</button></header>
      <div class="help-body">
        ${teacherMode ? '<section class="help-teacher-mode" role="status"><h3>Режим преподавателя</h3><p>Включён до обновления страницы. Чтобы выключить, снова удерживайте Alt и щёлкните заголовок справки.</p></section>' : ""}
        <section><h3>Настройка сети</h3><p>${classicMode ? "В меню «Модели»" : "На панели «Модель»"} выберите «Россыпь» для произвольной схемы или «Перцептрон» для обучения. Максимум — 32 нейрона. ${classicMode ? "Окна параметров и матрицы открываются через меню «Вид»." : "Параметры нейронов и матрица связей доступны на вкладках под графиком."} Строка матрицы — источник, столбец — приёмник.</p></section>
        <section><h3>Расчёт</h3><p>«Шаг» выполняет один расчёт, «Пуск» — непрерывный. P — вход нейрона, z — выход после порога и функции активации. Постоянные входы вводятся прямо в таблице векторов. Изменения действуют со следующего шага, без очистки графика.</p></section>
        <section><h3>Обучение</h3><p>Задайте образ, начальные веса, цели 0/1 и скорость обучения η. У выходов выберите знаковую функцию. «Одна коррекция» меняет веса для текущего образа. «Обучить образ» повторяет коррекции до верного ответа или ограничения числа попыток.${teacherMode ? " «Обучить весь набор» повторяет проходы по сохранённым образам." : ""} «Прервать обучение» сохраняет полученные веса. «Проверить весь набор» проверяет ответы без изменения весов. После обучения новому символу проверьте предыдущие: их ответы могут измениться.</p></section>
        <section><h3>Сохранение</h3><p>«Сохранить» и «Открыть»${classicMode ? " в меню «Файл»" : ""} записывают и загружают параметры, веса и учебный набор. После открытия время и график начинаются заново. Изменение топологии очищает веса и набор после подтверждения.</p></section>
      </div>
      <footer><span>Горячая клавиша: пробел — один шаг (если курсор не находится в поле ввода).</span><button class="primary-button" data-action="close-help">Понятно</button></footer>
    </section>
  </div>`;
}

function render(restorePreviousFocus = true): void {
  clearPendingInputRender();
  prepareStudioLayout();
  cancelPatternResize();
  if (classicMode) {
    cancelClassicDrag();
    cancelClassicSplitterDrag();
  }
  const previousFocus = restorePreviousFocus ? captureFocus() : null;
  const previousScroll = capturePanelScroll();
  selectNeuron(selectedNeuron);
  reconcileClassicWindowState();
  const tabs = renderTabs();
  const settings = classicMode ? renderClassicPanels()
    : `<section class="settings-panel" data-classic-tab="${activeTab}">
        ${tabs}
        <div class="tab-runtime-lock"><div class="tab-content" role="tabpanel">${renderActiveTab()}</div></div>
      </section>`;
  const floatingWindows = classicMode
    ? `<div class="classic-window-layer">${settings}${renderClassicNeuronWindows()}</div>`
    : settings;
  app.innerHTML = `
    ${renderHeader()}
    <main${classicMode ? ` data-status-visible="${classicView.status}"` : ""}>
      <div class="toast ${toastTone}" role="status" aria-live="polite" ${classicMode && !classicView.status ? "hidden" : ""}>${escapeHtml(toast)}</div>
      <section class="workspace"${classicMode ? ` data-neurons-visible="${classicView.neurons}" data-graph-visible="${classicView.graph}" style="--classic-neuron-pane-width:${classicNeuronFieldWidth}px"` : ""}>
        ${!classicMode || classicView.neurons ? renderNeuronField() : ""}${classicMode && classicView.neurons && classicView.graph ? renderClassicWorkspaceSplitter() : ""}${!classicMode || classicView.graph ? renderGraphPanel() : ""}
      </section>
      ${floatingWindows}
    </main>
    <footer class="app-footer"><span>NeuroNet Online · учебная программа</span><span>Данные остаются на этом компьютере</span></footer>
    ${renderHelp()}`;
  updateRuntimeLocks();
  restorePanelScroll(previousScroll);
  restoreFocus(previousFocus);
  if (studioMode) syncStudioTopologies(app, state);
  if (classicMode) {
    window.requestAnimationFrame(() => {
      resizeClassicWorkspace();
      if (studioMode || window.innerWidth > 820) setClassicNeuronFieldWidth(classicNeuronFieldPreferredWidth, false);
      clampClassicWindowPositions();
    });
  }
}

function resizeClassicWorkspace(): void {
  if (!classicMode) return;
  const headerHeight = app.querySelector<HTMLElement>(".classic-shell-header")?.getBoundingClientRect().height ?? 72;
  const statusHeight = classicView.status
    ? studioMode ? app.querySelector<HTMLElement>(".toast")?.getBoundingClientRect().height ?? 24 : 19
    : 0;
  app.querySelector<HTMLElement>(".workspace")?.style.setProperty("--classic-workspace-height", `${Math.max(180, window.innerHeight - headerHeight - statusHeight)}px`);
}

function renderRuntimePanels(): void {
  // Keep the pressed control alive through its native pointer/key click.
  if (inputPointerInteraction || runtimeKeyboardInteraction) return;
  const previousFocus = captureFocus();
  const previousScroll = capturePanelScroll();
  const workspace = app.querySelector<HTMLElement>(".workspace");
  if (workspace && classicMode) {
    const neuronPanel = workspace.querySelector<HTMLElement>(".neuron-panel");
    const graphPanel = workspace.querySelector<HTMLElement>(".graph-panel");
    if (neuronPanel) neuronPanel.outerHTML = renderNeuronField();
    if (graphPanel) graphPanel.outerHTML = renderGraphPanel();
  } else if (workspace) {
    workspace.innerHTML = `${renderNeuronField()}${renderGraphPanel()}`;
  }
  app.querySelectorAll<HTMLElement>(".classic-neuron-properties[data-neuron-window-index]").forEach((windowElement) => {
    const index = Number(windowElement.dataset.neuronWindowIndex);
    const neuron = state.neurons[index];
    if (!neuron) return;
    const inputImpact = neuron.potential;
    const inputElement = windowElement.querySelector<HTMLElement>('[data-classic-live="input-impact"]');
    const outputElement = windowElement.querySelector<HTMLElement>('[data-classic-live="output"]');
    const timeElement = windowElement.querySelector<HTMLElement>('[data-classic-live="time"]');
    if (inputElement) inputElement.textContent = fmt(inputImpact);
    if (outputElement) outputElement.textContent = fmt(neuron.output);
    if (timeElement) timeElement.textContent = fmt(state.time);
  });
  app.querySelectorAll<HTMLElement>(".settings-panel .tab-content").forEach(content => {
    const tab = content.closest<HTMLElement>(".settings-panel")?.dataset.classicTab as Tab;
    // Keep the actual editing node alive, including incomplete numeric drafts,
    // caret, IME composition and an open native select. Other windows stay live.
    const focused = document.activeElement;
    if (focused instanceof HTMLElement && content.contains(focused) && focused.matches("input:enabled, select:enabled, textarea:enabled")) {
      content.querySelectorAll<HTMLElement>(".neuron-row[data-index]").forEach(row => {
        const neuron = state.neurons[Number(row.dataset.index)];
        if (!neuron) return;
        const potential = row.querySelector<HTMLElement>("[data-neuron-potential]");
        const output = row.querySelector<HTMLElement>("[data-neuron-output]");
        if (potential) potential.textContent = `P ${fmt(neuron.potential)}`;
        if (output) output.textContent = `z ${fmt(neuron.output)}`;
      });
      const selectedOutput = content.querySelector<HTMLElement>("[data-editor-output]");
      if (selectedOutput) selectedOutput.textContent = `z = ${fmt(state.neurons[selectedNeuron]!.output)}`;
      const trainingStatus = content.querySelector<HTMLElement>(".training-state");
      if (trainingStatus) trainingStatus.outerHTML = renderTrainingStatus();
      content.querySelectorAll<HTMLElement>("[data-vector-neuron]").forEach(row => {
        const neuron = state.neurons[Number(row.dataset.vectorNeuron)];
        if (!neuron) return;
        const output = row.querySelector<HTMLElement>("[data-vector-output]");
        if (output) output.textContent = fmt(neuron.output);
        const potential = row.querySelector<HTMLElement>("[data-vector-potential]");
        if (potential) potential.textContent = fmt(neuron.potential);
      });
      return;
    }
    content.innerHTML = renderActiveTab(tab);
  });
  updateRuntimeLocks();
  restorePanelScroll(previousScroll);
  restoreFocus(previousFocus);
  syncStudioLayout();
  if (studioMode) syncStudioTopologies(app, state);
  const readout = app.querySelector<HTMLElement>(".runtime-readout");
  if (readout) {
    readout.innerHTML = `<span class="runtime-dot ${runtimeMode}"></span>
      <strong>${runtimeMode === "running" ? "Выполняется" : runtimeMode === "training" ? "Обучение" : "Остановлено"}</strong>
      <span>шаг ${state.step}</span><span>t = ${fmt(state.time)}</span>`;
  }
  const toastElement = app.querySelector<HTMLElement>(".toast");
  if (toastElement) {
    toastElement.className = `toast ${toastTone}`;
    toastElement.textContent = toast;
  }
}

function openHelp(): void {
  helpReturnFocus = captureFocus();
  helpOpen = true;
  render(false);
  app.querySelector<HTMLElement>('.help-modal [data-action="close-help"]')?.focus();
}

function closeHelp(): void {
  const returnTarget = helpReturnFocus;
  helpOpen = false;
  helpReturnFocus = null;
  render(false);
  restoreFocus(returnTarget);
}

function openTeachingScenarios(): void {
  if (!teacherMode) return;
  helpOpen = false;
  helpReturnFocus = null;
  activeTab = "labs";
  if (classicMode) openClassicSettings("labs");
  render(false);
  app.querySelector<HTMLElement>('[data-classic-tab="labs"] select')?.focus({ preventScroll: true });
}

function toggleTeacherMode(): void {
  teacherMode = !teacherMode;
  if (teacherMode) {
    setToast("Режим преподавателя включён до обновления страницы.");
    openTeachingScenarios();
    return;
  }
  const wasTrainingSet = runtimeMode === "training" && trainingTarget === "examples";
  if (wasTrainingSet) stopAll();
  closeClassicPanel("labs");
  if (activeTab === "labs") activeTab = state.model === "perceptron" ? "perceptron" : "neurons";
  setToast(wasTrainingSet
    ? "Режим преподавателя выключен. Обучение набора остановлено; полученные веса сохранены."
    : "Режим преподавателя выключен.");
  closeHelp();
}

function replaceWith(next: NetworkState, message: string, tone: typeof toastTone = "success"): void {
  runtimeNeedsStep = true;
  examplesReport = null;
  examplesEpochs = 0;
  exampleNameDraft = "";
  stopAll();
  state = next;
  selectNeuron(0);
  setToast(message, tone);
  render();
}

function confirmReplace(message: string): boolean {
  const blank = createNetwork(16);
  const hasConfiguredWork = JSON.stringify(resetRuntime(state)) !== JSON.stringify(blank);
  return !hasConfiguredWork && state.step === 0 && state.history.length === 0
    ? true
    : window.confirm(message);
}

function downloadProject(): void {
  const data = serializeState(state);
  const blob = new Blob([data], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const safeName = state.title === "Без имени" ? "NeuroNet" : state.title.replace(/[<>:"/\\|?*\u0000-\u001F]+/g, "-").trim() || "NeuroNet";
  anchor.href = url;
  anchor.download = `${safeName}.neuronet.json`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  setToast("Проект сохранён.", "success");
  render();
}

function doStep(): void {
  try {
    state = runStep(state).state;
    runtimeNeedsStep = false;
    setToast(`Выполнен шаг ${state.step}.`, "success");
  } catch (error) {
    stopAll();
    setToast(error instanceof Error ? error.message : "Не удалось вычислить шаг. Проверьте параметры сети.", "warning");
  }
  render();
}

function startRun(): void {
  if (runtimeMode === "running") {
    stopAll();
    setToast("Выполнение приостановлено.");
    render();
    return;
  }
  stopAll();
  runtimeMode = "running";
  setToast("Модель выполняется непрерывно. Нажмите «Пауза» или «Стоп».");
  render();
  runTimer = window.setInterval(() => {
    try {
      state = runStep(state).state;
      runtimeNeedsStep = false;
      renderRuntimePanels();
    } catch (error) {
      stopAll();
      setToast(error instanceof Error ? error.message : "Вычисление остановлено: проверьте параметры сети.", "warning");
      render();
    }
  }, 320);
}

function doTrainingStep(): boolean {
  const issue = trainingSupportIssue(state);
  if (issue) {
    setToast(issue, "warning");
    return false;
  }
  const result = trainStep(state);
  state = result.state;
  if (Number.isFinite(result.error)) runtimeNeedsStep = false;
  lastChangedWeights = result.changedWeights;
  examplesReport = null;
  if ((state.perceptron.trainingExamples?.length ?? 0) > 0 && !trainingExamplesIssue(state)) {
    try { examplesReport = evaluateExamples(state); } catch { /* Keep an invalid/overflowing set unverified. */ }
  }
  if (result.blockedReason) {
    stopAll();
    setToast(result.blockedReason, "warning");
  } else if (result.learned) {
    setToast(`Цель достигнута за ${state.perceptron.trainingSteps} шаг(а/ов) обучения.`, "success");
  } else {
    setToast(`Коррекция ${state.perceptron.trainingSteps}; ошибка ${fmt(result.error, 6)}.`);
  }
  return result.learned;
}

function startTraining(): void {
  if (runtimeMode === "training") return;
  const issue = trainingSupportIssue(state);
  if (issue) {
    setToast(issue, "warning");
    render();
    return;
  }
  stopAll();
  runtimeMode = "training";
  trainingTarget = "current";
  trainingAttempts = 0;
  setToast("Обучение текущего образа запущено. Максимум — 1000 коррекций.");
  render();
  trainingTimer = window.setInterval(() => {
    const issueDuringTraining = trainingSupportIssue(state);
    if (issueDuringTraining) {
      stopAll();
      setToast(`Обучение остановлено: ${issueDuringTraining}`, "warning");
      render();
      return;
    }
    const learned = doTrainingStep();
    trainingAttempts += 1;
    const reachedLimit = trainingAttempts >= 1000;
    if (learned || reachedLimit) {
      stopAll();
      if (reachedLimit && !learned) {
        setToast("После 1000 коррекций цель не достигнута. Проверьте вход, цель, порог и веса.", "warning");
      }
    }
    if (learned || reachedLimit || runtimeMode === "stopped") render();
    else renderRuntimePanels();
  }, 70);
}

function checkExamples(): void {
  try {
    examplesReport = evaluateExamples(state);
    setToast(`Проверка без изменения весов: верно ${examplesReport.correct} / ${examplesReport.total}.`, examplesReport.correct === examplesReport.total ? "success" : "warning");
  } catch (error) {
    examplesReport = null;
    setToast(error instanceof Error ? error.message : "Не удалось проверить набор.", "warning");
  }
}

function doExamplesEpoch(): boolean {
  const result = trainEpoch(state);
  if (result.blockedReason || !result.report) {
    stopAll();
    examplesReport = null;
    setToast(result.blockedReason ?? "Не удалось обучить набор.", "warning");
    return false;
  }
  state = result.state;
  examplesReport = result.report;
  examplesEpochs += 1;
  lastChangedWeights = result.changedWeights;
  runtimeNeedsStep = false;
  const complete = examplesReport.correct === examplesReport.total;
  setToast(`Проход ${examplesEpochs}: верно ${examplesReport.correct} / ${examplesReport.total} на окончательных весах.${complete ? " Весь набор распознан." : ""}`, complete ? "success" : "info");
  return complete;
}

function startExamplesTraining(): void {
  if (!teacherMode) return;
  const issue = trainingExamplesIssue(state) ?? trainingSupportIssue(state);
  if (issue) { setToast(issue, "warning"); render(); return; }
  stopAll();
  trainingTarget = "examples";
  examplesEpochs = 0;
  trainingAttempts = 0;
  checkExamples();
  if (examplesReport && examplesReport.correct === examplesReport.total) { render(); return; }
  if (!examplesReport) { render(); return; }
  runtimeMode = "training";
  setToast("Обучение всего набора. После каждого прохода проверяются все сохранённые образы; максимум 500 проходов.");
  render();
  trainingTimer = window.setInterval(() => {
    const complete = doExamplesEpoch();
    trainingAttempts += 1;
    const limit = trainingAttempts >= 500;
    if (complete || limit) {
      stopAll();
      if (limit && !complete) setToast(`После 500 проходов верно ${examplesReport?.correct ?? 0} / ${examplesReport?.total ?? 0}. Сходимость не достигнута: проверьте разделимость образов, входные данные и пороги.`, "warning");
    }
    if (complete || limit || runtimeMode === "stopped") render();
    else renderRuntimePanels();
  }, 70);
}

function addCurrentExample(): void {
  if (state.neurons.slice(0, state.perceptron.inputs).some(neuron => neuron.inputKind !== "constant")) {
    setToast("Учебный набор хранит статические образы. Задайте постоянные входы или введите двоичный образ в редакторе.", "warning");
    render(); return;
  }
  const examples = state.perceptron.trainingExamples ?? [];
  const inputs = state.neurons.slice(0, state.perceptron.inputs).map(neuron => neuron.constantInput);
  const existing = examples.findIndex(example => example.inputs.every((value, index) => value === inputs[index]));
  if (existing < 0 && examples.length >= MAX_TRAINING_EXAMPLES) {
    setToast(`В наборе допускается не более ${MAX_TRAINING_EXAMPLES} образов.`, "warning"); render(); return;
  }
  if (existing >= 0 && !window.confirm(`Такой вход уже сохранён как «${examples[existing]!.name}». Заменить его цели и название текущими?`)) return;
  const example = { name: exampleNameDraft.trim() || (existing >= 0 ? examples[existing]!.name : `Образ ${examples.length + 1}`), inputs, desired: [...state.perceptron.desired] };
  const candidate = [...examples];
  if (existing >= 0) candidate[existing] = example;
  else candidate.push(example);
  const issue = trainingExamplesIssue(state, candidate);
  if (issue) { setToast(issue, "warning"); render(); return; }
  state.perceptron.trainingExamples = candidate;
  examplesReport = null;
  examplesEpochs = 0;
  exampleNameDraft = "";
  setToast(`Образ «${example.name}» ${existing >= 0 ? "обновлён" : "добавлен"}. В наборе ${candidate.length}; веса не изменились.`, "success");
  render();
}

function zeroWeights(): void {
  if (!window.confirm("Обнулить все веса текущей модели?")) return;
  state.weights = state.weights.map((row) => row.map(() => 0));
  parametersChanged();
  setToast("Матрица связей обнулена.", "warning");
  render();
}

function parseField(target: HTMLInputElement | HTMLSelectElement): number {
  return finite(target.value, 0);
}

function closeClassicMenus(except?: HTMLDetailsElement): void {
  if (!classicMode) return;
  app.querySelectorAll<HTMLDetailsElement>(".classic-menu[open]").forEach((menu) => {
    if (menu !== except) menu.open = false;
  });
}

function setClassicWindowPosition(key: ClassicWindowKey, position: ClassicPosition, element: HTMLElement): void {
  if (key.startsWith("panel:")) {
    classicPanels.set(key.slice(6) as Tab, position);
  } else {
    const index = Number(key.slice("neuron:".length));
    const windowState = classicNeuronWindows.find((item) => item.index === index);
    if (windowState) {
      windowState.x = position.x;
      windowState.y = position.y;
    }
  }
  element.style.left = `${position.x}px`;
  element.style.top = `${position.y}px`;
  element.style.transform = "none";
}

function clampClassicPosition(element: HTMLElement, x: number, y: number): ClassicPosition {
  const rect = element.getBoundingClientRect();
  const headerBottom = app.querySelector<HTMLElement>(".classic-shell-header")?.getBoundingClientRect().bottom ?? 0;
  const statusTop = classicView.status ? (app.querySelector<HTMLElement>(".toast")?.getBoundingClientRect().top ?? window.innerHeight) : window.innerHeight;
  const minX = 4;
  const minY = Math.max(4, headerBottom + 4);
  const maxX = Math.max(minX, window.innerWidth - rect.width - 4);
  const maxY = Math.max(minY, statusTop - rect.height - 4);
  return {
    x: Math.max(minX, Math.min(maxX, x)),
    y: Math.max(minY, Math.min(maxY, y)),
  };
}

function clampClassicWindowPositions(): void {
  if (!classicMode) return;
  app.querySelectorAll<HTMLElement>("[data-classic-window-key]").forEach((element) => {
    const key = element.dataset.classicWindowKey as ClassicWindowKey;
    const position = key.startsWith("panel:")
      ? classicPanels.get(key.slice(6) as Tab)
      : classicNeuronWindows.find((item) => classicNeuronWindowKey(item.index) === key);
    if (!position) return;
    setClassicWindowPosition(key, clampClassicPosition(element, position.x, position.y), element);
  });
  syncClassicWindowLayers();
}

function cancelClassicDrag(): void {
  if (!classicDragState) return;
  const { handle, pointerId } = classicDragState;
  classicDragState = null;
  if (handle.hasPointerCapture(pointerId)) handle.releasePointerCapture(pointerId);
}

function cancelClassicSplitterDrag(): void {
  if (!classicSplitterDragState) return;
  const { handle, pointerId } = classicSplitterDragState;
  classicSplitterDragState = null;
  delete handle.dataset.dragging;
  if (handle.hasPointerCapture(pointerId)) handle.releasePointerCapture(pointerId);
}

function finishClassicPointerInteraction(event: PointerEvent): void {
  if (classicDragState?.pointerId === event.pointerId) cancelClassicDrag();
  if (classicSplitterDragState?.pointerId === event.pointerId) cancelClassicSplitterDrag();
  if (patternResizeState?.pointerId === event.pointerId) cancelPatternResize();
}

window.addEventListener("pointerdown", (event) => {
  if (event.button === 0 && event.isPrimary) inputPointerInteraction = true;
}, true);
window.addEventListener("pointerup", finishInputPointerInteraction);
window.addEventListener("pointercancel", finishInputPointerInteraction);
window.addEventListener("blur", finishInputPointerInteraction);

function releaseRuntimeKeyboardInteraction(): void {
  if (runtimeKeyboardReleaseTimer !== undefined) window.clearTimeout(runtimeKeyboardReleaseTimer);
  // Space activates a button after keyup dispatch; retain its DOM until then.
  runtimeKeyboardReleaseTimer = window.setTimeout(() => {
    runtimeKeyboardInteraction = false;
    runtimeKeyboardReleaseTimer = undefined;
  }, 0);
}

window.addEventListener("keydown", (event) => {
  if ((event.key === " " || event.key === "Enter") && event.target instanceof Element && event.target.closest("button")) {
    if (runtimeKeyboardReleaseTimer !== undefined) window.clearTimeout(runtimeKeyboardReleaseTimer);
    runtimeKeyboardReleaseTimer = undefined;
    runtimeKeyboardInteraction = true;
  }
}, true);
window.addEventListener("keyup", (event) => {
  if (event.key === " " || event.key === "Enter") releaseRuntimeKeyboardInteraction();
});
window.addEventListener("blur", releaseRuntimeKeyboardInteraction);

app.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || !event.isPrimary) return;
  const clicked = event.target as HTMLElement;
  const patternHandle = clicked.closest<HTMLButtonElement>("[data-pattern-resize]");
  if (patternHandle) {
    if (patternHandle.disabled || classicDragState || classicSplitterDragState || patternResizeState) return;
    patternResizeState = {
      pointerId: event.pointerId,
      handle: patternHandle,
      originColumns: patternColumns(),
      pointerX: event.clientX,
    };
    patternHandle.focus({ preventScroll: true });
    patternHandle.dataset.dragging = "true";
    patternHandle.setPointerCapture(event.pointerId);
    event.preventDefault();
    return;
  }
  if (!classicMode) return;
  const splitter = clicked.closest<HTMLElement>(".classic-workspace-splitter");
  if (splitter) {
    if (classicDragState || classicSplitterDragState || window.innerWidth <= 820) return;
    classicSplitterDragState = {
      pointerId: event.pointerId,
      handle: splitter,
      originWidth: classicNeuronFieldWidth,
      pointerX: event.clientX,
    };
    splitter.dataset.dragging = "true";
    splitter.setPointerCapture(event.pointerId);
    event.preventDefault();
    return;
  }
  const windowElement = clicked.closest<HTMLElement>("[data-classic-window-key]");
  if (!windowElement) return;
  const key = windowElement.dataset.classicWindowKey as ClassicWindowKey;
  bringClassicWindowToFront(key);

  const handle = clicked.closest<HTMLElement>(".classic-dialog-title, .classic-neuron-window-title");
  if (!handle || clicked.closest("button") || classicDragState || classicSplitterDragState) return;
  const rect = windowElement.getBoundingClientRect();
  const origin = clampClassicPosition(windowElement, rect.left, rect.top);
  setClassicWindowPosition(key, origin, windowElement);
  classicDragState = {
    key,
    pointerId: event.pointerId,
    windowElement,
    handle,
    originX: origin.x,
    originY: origin.y,
    pointerX: event.clientX,
    pointerY: event.clientY,
  };
  handle.setPointerCapture(event.pointerId);
  event.preventDefault();
});

app.addEventListener("focusin", (event) => {
  if (!classicMode) return;
  const windowElement = (event.target as HTMLElement).closest<HTMLElement>("[data-classic-window-key]");
  if (!windowElement) return;
  const key = windowElement.dataset.classicWindowKey as ClassicWindowKey;
  if (activeClassicWindow() !== key) bringClassicWindowToFront(key);
});

window.addEventListener("pointermove", (event) => {
  const patternDrag = patternResizeState;
  if (patternDrag && event.pointerId === patternDrag.pointerId) {
    const columnWidth = studioMode ? STUDIO_COLUMN_PITCH : classicMode ? 48 : 56;
    setPatternColumns(patternDrag.originColumns + (event.clientX - patternDrag.pointerX) / columnWidth);
    if (event.cancelable) event.preventDefault();
    return;
  }
  const splitterDrag = classicSplitterDragState;
  if (splitterDrag && event.pointerId === splitterDrag.pointerId) {
    setClassicNeuronFieldWidth(splitterDrag.originWidth + event.clientX - splitterDrag.pointerX);
    if (event.cancelable) event.preventDefault();
    return;
  }
  const drag = classicDragState;
  if (!drag || event.pointerId !== drag.pointerId) return;
  const next = clampClassicPosition(
    drag.windowElement,
    drag.originX + event.clientX - drag.pointerX,
    drag.originY + event.clientY - drag.pointerY,
  );
  setClassicWindowPosition(drag.key, next, drag.windowElement);
  if (event.cancelable) event.preventDefault();
});

window.addEventListener("pointerup", finishClassicPointerInteraction);
window.addEventListener("pointercancel", finishClassicPointerInteraction);
app.addEventListener("lostpointercapture", finishClassicPointerInteraction);
window.addEventListener("blur", () => {
  cancelClassicDrag();
  cancelClassicSplitterDrag();
  cancelPatternResize();
});
window.addEventListener("resize", () => window.requestAnimationFrame(() => {
  resizeClassicWorkspace();
  if (studioMode || window.innerWidth > 820) setClassicNeuronFieldWidth(classicNeuronFieldPreferredWidth, false);
  clampClassicWindowPositions();
}));
// Update only this control: fullscreen must not replace editors or reset a run.
document.addEventListener("fullscreenchange", syncFullscreenControl);

app.addEventListener("dblclick", (event) => {
  if (!classicMode || !(event.target as HTMLElement).closest(".classic-workspace-splitter")) return;
  if (studioMode) { rememberStudioColumns(4); syncStudioLayout(); }
  else setClassicNeuronFieldWidth(CLASSIC_NEURON_FIELD_DEFAULT_WIDTH);
});

app.addEventListener("click", (event) => {
  const clicked = event.target as HTMLElement;
  if (helpOpen && event.altKey && clicked.closest("#help-title")) {
    event.preventDefault();
    toggleTeacherMode();
    return;
  }
  if (classicMode) {
    const summary = clicked.closest<HTMLElement>(".classic-menu > summary");
    if (summary) {
      closeClassicMenus(summary.parentElement as HTMLDetailsElement);
      return;
    }
    closeClassicMenus();
  }
  const button = clicked.closest<HTMLElement>("[data-action]");
  if (!button) return;
  if (button instanceof HTMLButtonElement && button.disabled) return;
  const action = button.dataset.action;
  if (!teacherMode && (action === "examples-start" || action?.startsWith("preset-") ||
      ((action === "tab" || action === "classic-toggle-panel") && button.dataset.tab === "labs"))) return;
  if (action === "toggle-fullscreen") {
    void toggleFullscreen();
    return;
  }
  if (inputRenderPending) render();
  if (action === "close-help" && clicked.closest("[data-modal]") && button === clicked.closest(".modal-backdrop")) return;

  if (action === "tab") {
    activeTab = button.dataset.tab as Tab;
    if (classicMode) openClassicSettings();
    render();
  } else if (action === "classic-toggle-panel") {
    const tab = button.dataset.tab as Tab;
    if (tab === "perceptron" && state.model !== "perceptron") return;
    if (classicPanels.has(tab)) closeClassicPanel(tab);
    else openClassicSettings(tab);
    render();
  } else if (action === "classic-toggle-view") {
    const key = button.dataset.view as keyof typeof classicView;
    classicView[key] = !classicView[key];
    render();
  } else if (action === "classic-close") {
    closeClassicPanel(button.dataset.tab as Tab);
    render();
  } else if (action === "classic-close-neuron") {
    const index = Number(button.dataset.index);
    classicNeuronWindows = classicNeuronWindows.filter((item) => item.index !== index);
    removeClassicWindowFromStack(classicNeuronWindowKey(index));
    render();
  } else if (action === "select-neuron") {
    const index = Number(button.dataset.index);
    selectNeuron(index);
    if (classicMode) {
      if (button.dataset.neuronOrigin === "field") openClassicNeuronWindow(index);
    } else {
      activeTab = "neurons";
    }
    render();
  } else if (action === "open-graph-settings") {
    activeTab = "graph";
    if (classicMode) openClassicSettings();
    render();
  } else if (action === "step") {
    doStep();
  } else if (action === "run") {
    startRun();
  } else if (action === "stop") {
    stopAll();
    setToast("Процесс остановлен.");
    render();
  } else if (action === "reset-runtime") {
    replaceWith(resetRuntime(state), "Вычисленные значения, время и график сброшены.");
  } else if (action === "zero-inputs") {
    stopAll();
    state = zeroInputs(state);
    parametersChanged();
    setToast("Входы, выходы, время и график обнулены. Схема, веса и параметры сохранены.");
    render();
  } else if (action === "new-project") {
    if (confirmReplace("Очистить текущий проект и создать пустую сеть?")) {
      replaceWith(createNetwork(16), "Создана пустая сеть из 16 нейронов.");
      activeTab = "neurons";
      classicPanels.clear();
      classicNeuronWindows = [];
      classicWindowStack = [];
      render();
    }
  } else if (action === "apply-size") {
    const field = document.querySelector<HTMLInputElement>("#network-size");
    const requested = Math.round(finite(field?.value, state.neurons.length));
    if (requested < MIN_NEURONS || requested > MAX_NEURONS) {
      setToast(`Допустимо от ${MIN_NEURONS} до ${MAX_NEURONS} нейронов.`, "warning");
      render();
    } else if (requested < state.neurons.length && !window.confirm(`Удалить нейроны H${requested + 1}…H${state.neurons.length} и связанные с ними веса?`)) {
      return;
    } else {
      replaceWith(resizeNetwork(state, requested), `Размер сети изменён: ${requested} нейронов.`);
    }
  } else if (action === "open-perceptron") {
    if (state.model !== "perceptron") return;
    activeTab = "perceptron";
    if (classicMode) openClassicSettings("perceptron");
    render();
  } else if (action === "switch-model") {
    const model = button.dataset.model as NetworkState["model"];
    if (model === state.model) {
      if (model === "perceptron") {
        activeTab = "perceptron";
        openClassicSettings("perceptron");
        render();
      }
      return;
    }
    if (model !== state.model && window.confirm("Смена модели очищает матрицу, входы, учебный набор и результаты. Продолжить?")) {
      replaceWith(switchModel(state, model), model === "perceptron" ? "Выбрана модель «Перцептрон»." : "Выбрана модель «Россыпь».");
      activeTab = model === "perceptron" ? "perceptron" : "neurons";
      if (model === "perceptron") openClassicSettings("perceptron");
      render();
    }
  } else if (action === "preset-single") {
    const variantNumber = Math.round(finite(document.querySelector<HTMLSelectElement>("#lab-single-variant")?.value, 1));
    const variant = LAB1_VARIANTS[Math.max(0, Math.min(LAB1_VARIANTS.length - 1, variantNumber - 1))]!;
    if (confirmReplace("Загрузить пресет и заменить текущий проект?")) {
      const next = lab1SingleNeuronPreset(button.dataset.activation as ActivationKind, variant.q21);
      next.title += `, вариант ${variantNumber}`;
      replaceWith(next, `Загружена схема H1 → H2, вариант ${variantNumber}. Пороги Q₂: ${fmt(variant.q21)} / ${fmt(variant.q22)} / ${fmt(variant.q23)}; сейчас установлен первый.`);
      activeTab = "neurons";
      openClassicSettings("neurons");
      selectNeuron(1);
      render();
    }
  } else if (action === "preset-summing") {
    const variant = finite(document.querySelector<HTMLSelectElement>("#lab-variant")?.value, 1);
    const activation = document.querySelector<HTMLSelectElement>("#lab-sum-activation")?.value as ActivationKind;
    if (confirmReplace("Загрузить пресет и заменить текущий проект?")) {
      replaceWith(lab1SummingPreset(variant, activation), `Задание 4, вариант ${variant}: исследуйте Q₄ = 0 / 0,5 / −0,2; сейчас установлен первый.`);
      activeTab = "neurons";
      openClassicSettings("neurons");
      selectNeuron(3);
      render();
    }
  } else if (action === "preset-lab2") {
    if (confirmReplace("Загрузить пресет ЛР2 и заменить текущий проект?")) {
      replaceWith(lab2PerceptronPreset(), "Загружен пресет ЛР2: H1…H14 → H15. Выход H15 сам обучает входящие веса.");
      activeTab = "perceptron";
      openClassicSettings("perceptron");
      selectNeuron(14);
      render();
    }
  } else if (action === "apply-architecture") {
    const inputs = finite(document.querySelector<HTMLInputElement>("#arch-inputs")?.value, state.perceptron.inputs);
    const outputs = finite(document.querySelector<HTMLInputElement>("#arch-outputs")?.value, state.perceptron.outputs);
    if (!Number.isInteger(inputs) || !Number.isInteger(outputs) || inputs + outputs > MAX_NEURONS || inputs < 1 || outputs < 1) {
      setToast(`Нужны целые количества: минимум 1 вход и 1 выход; всего не более ${MAX_NEURONS}.`, "warning");
      render();
    } else if (state.model === "perceptron" && inputs === state.perceptron.inputs && outputs === state.perceptron.outputs) {
      setToast("Топология не изменилась: веса, параметры и учебный набор сохранены. Раскладку рисунка можно изменить, потянув за его нижний правый уголок.");
      render();
    } else if (window.confirm("Изменение топологии обнулит веса, очистит учебный набор и восстановит стандартные пороги и функции нейронов. Для изменения только раскладки потяните за нижний правый уголок рисунка. Продолжить?")) {
      replaceWith(configurePerceptron(state, inputs, outputs), "Топология применена. Задайте входы, начальные веса и цели в окне «Перцептрон».");
      activeTab = "perceptron";
    }
  } else if (action === "training-step") {
    doTrainingStep();
    render();
  } else if (action === "training-start") {
    startTraining();
  } else if (action === "example-add") {
    addCurrentExample();
  } else if (action === "example-load") {
    const example = state.perceptron.trainingExamples?.[Number(button.dataset.index)];
    if (example) {
      example.inputs.forEach((value, index) => { state.neurons[index]!.constantInput = value; state.neurons[index]!.inputKind = "constant"; });
      state.perceptron.desired = [...example.desired];
      exampleNameDraft = example.name;
      parametersChanged(false);
      setToast(`Выбран образ «${example.name}». Веса сохранены; выполните шаг для его ответа.`);
      render();
    }
  } else if (action === "example-remove") {
    const index = Number(button.dataset.index);
    const example = state.perceptron.trainingExamples?.[index];
    if (example && window.confirm(`Удалить образ «${example.name}» из учебного набора? Веса сохранятся.`)) {
      state.perceptron.trainingExamples!.splice(index, 1);
      examplesReport = null;
      examplesEpochs = 0;
      setToast("Образ удалён из набора. Веса не изменились."); render();
    }
  } else if (action === "examples-check") {
    checkExamples(); render();
  } else if (action === "examples-start") {
    startExamplesTraining();
  } else if (action === "toggle-bit") {
    const index = Number(button.dataset.index);
    const neuron = state.neurons[index];
    if (neuron) { neuron.constantInput = neuron.constantInput >= 0.5 ? 0 : 1; neuron.inputKind = "constant"; }
    parametersChanged();
    setToast(`Изменён компонент x${index + 1}.`);
    render();
  } else if (action === "zero-weights") {
    zeroWeights();
  } else if (action === "graph-all") {
    state.graph.selected = state.neurons.map((_, index) => index);
    render();
  } else if (action === "graph-none") {
    state.graph.selected = [];
    render();
  } else if (action === "clear-graph") {
    state.history = [];
    setToast("История графика очищена.");
    render();
  } else if (action === "export") {
    downloadProject();
  } else if (action === "import") {
    document.querySelector<HTMLInputElement>("#import-file")?.click();
  } else if (action === "print") {
    window.print();
  } else if (action === "help") {
    openHelp();
  } else if (action === "close-help") {
    closeHelp();
  }
});

app.addEventListener("change", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement)) return;
  if (target instanceof HTMLInputElement && target.type === "number" && (!Number.isFinite(target.valueAsNumber) || !target.validity.valid)) {
    setToast("Введите допустимое число в поле; прежнее значение сохранено.", "warning");
    renderAfterInputChange();
    return;
  }

  if (target.id === "import-file" && target instanceof HTMLInputElement) {
    const file = target.files?.[0];
    if (!file) return;
    try {
      const parsed: unknown = JSON.parse(await file.text());
      const loaded = deserializeState(parsed);
      replaceWith(resetRuntime(loaded.state), `Проект открыт.${loaded.notice ? ` ${loaded.notice}` : ""}`, loaded.notice ? "info" : "success");
    } catch (error) {
      setToast(`Не удалось открыть файл: ${error instanceof Error ? error.message : "неизвестная ошибка"}.`, "warning");
      render();
    }
    return;
  }

  if (target.matches("[data-example-name]")) {
    exampleNameDraft = target.value.slice(0, 80);
    return;
  }
  if (target.matches("[data-pattern-string]")) {
    if (!/^[01]+$/.test(target.value) || target.value.length !== state.perceptron.inputs) {
      setToast(`Введите ровно ${state.perceptron.inputs} нулей и единиц, по строкам рисунка.`, "warning");
    } else {
      [...target.value].forEach((bit, index) => { state.neurons[index]!.constantInput = Number(bit); state.neurons[index]!.inputKind = "constant"; });
      parametersChanged();
      setToast("Входной образ применён; обученные веса сохранены.");
    }
    renderAfterInputChange();
    return;
  }

  const neuronField = target.dataset.neuronField;
  if (neuronField) {
    const neuron = state.neurons[selectedNeuron]!;
    if (neuronField === "role") neuron.role = target.value as NeuronRole;
    else if (neuronField === "activation") neuron.activation = target.value as ActivationKind;
    else if (neuronField === "inputKind") neuron.inputKind = target.value as InputKind;
    else if (neuronField === "threshold") neuron.threshold = parseField(target);
    else if (neuronField === "constantInput") neuron.constantInput = parseField(target);
    else if (neuronField === "linear.slope") neuron.linear.slope = parseField(target);
    else if (neuronField === "linear.intercept") neuron.linear.intercept = parseField(target);
    else if (neuronField === "harmonic.amplitude") neuron.harmonic.amplitude = parseField(target);
    else if (neuronField === "harmonic.omega") neuron.harmonic.omega = parseField(target);
    else if (neuronField === "harmonic.phase") neuron.harmonic.phase = parseField(target);
    else if (neuronField === "harmonic.deltaTime") neuron.harmonic.deltaTime = Math.max(0.000001, parseField(target));
    parametersChanged();
    setToast(neuronField === "harmonic.deltaTime"
      ? `Шаг Δt изменён. Далее используется новая сетка t=k·Δt; номер шага и прежний график сохранены.`
      : `Параметры H${neuron.id} приняты для следующего шага. Время и график сохранены.`);
    renderAfterInputChange();
    return;
  }

  if (target.matches("[data-weight]")) {
    const source = Number(target.dataset.source);
    const receiver = Number(target.dataset.target);
    if (state.weights[source]?.[receiver] !== undefined) {
      state.weights[source]![receiver] = parseField(target);
      parametersChanged();
      setToast(`Вес w${source + 1},${receiver + 1} принят для следующего шага. Время и график сохранены.`);
      renderAfterInputChange();
    }
    return;
  }

  if (target.matches("[data-vector-input]")) {
    const index = Number(target.dataset.index);
    if (state.neurons[index]) state.neurons[index]!.constantInput = parseField(target);
    parametersChanged();
    setToast(`Вход x${index + 1} принят для следующего шага. Время и график сохранены.`);
    renderAfterInputChange();
    return;
  }

  if (target.matches("[data-perceptron-field]")) {
    state.perceptron.learningRate = Math.max(0.000001, parseField(target));
    state.perceptron.trained = false;
    setToast("Скорость обучения изменена.");
    renderAfterInputChange();
    return;
  }

  if (target.matches("[data-desired]")) {
    const index = Number(target.dataset.index);
    state.perceptron.desired[index] = parseField(target) >= 0.5 ? 1 : 0;
    state.perceptron.trained = false;
    setToast(`Цель выхода ${index + 1} изменена.`);
    renderAfterInputChange();
    return;
  }

  if (target.matches("[data-graph-neuron]") && target instanceof HTMLInputElement) {
    const index = Number(target.dataset.index);
    state.graph.selected = target.checked
      ? [...new Set([...state.graph.selected, index])].sort((a, b) => a - b)
      : state.graph.selected.filter((item) => item !== index);
    renderAfterInputChange();
    return;
  }

  if (target.matches("[data-graph-kind]")) {
    state.graph.kind = target.value as GraphKind;
    renderAfterInputChange();
    return;
  }

  if (target.matches("[data-graph-max]")) {
    state.graph.maxSamples = Math.max(20, Math.min(500, Math.round(parseField(target))));
    if (state.history.length > state.graph.maxSamples) {
      state.history.splice(0, state.history.length - state.graph.maxSamples);
    }
    renderAfterInputChange();
  }
});

window.addEventListener("keydown", (event) => {
  // Escape first belongs to the browser's fullscreen UI, not the floating windows.
  if (event.key === "Escape" && document.fullscreenElement) return;
  if (helpOpen) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeHelp();
      return;
    }
    if (event.key === "Tab") {
      const modal = app.querySelector<HTMLElement>("[data-modal]");
      const focusable = modal
        ? Array.from(modal.querySelectorAll<HTMLElement>('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'))
            .filter((item) => !item.hasAttribute("disabled"))
        : [];
      if (focusable.length > 0) {
        const first = focusable[0]!;
        const last = focusable[focusable.length - 1]!;
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        } else if (!modal?.contains(document.activeElement)) {
          event.preventDefault();
          first.focus();
        }
      }
    }
    return;
  }
  if (classicMode && event.key === "Escape") {
    const openMenus = app.querySelectorAll<HTMLDetailsElement>(".classic-menu[open]");
    if (openMenus.length > 0) {
      event.preventDefault();
      closeClassicMenus();
      return;
    }
    const key = activeClassicWindow();
    if (key) {
      event.preventDefault();
      if (key.startsWith("panel:")) closeClassicPanel(key.slice(6) as Tab);
      else {
        classicNeuronWindows = classicNeuronWindows.filter(item => classicNeuronWindowKey(item.index) !== key);
        removeClassicWindowFromStack(key);
      }
      render();
      return;
    }
  }
  const element = event.target as HTMLElement;
  if (event.key === "Enter" && element instanceof HTMLInputElement) {
    event.preventDefault();
    element.blur();
    return;
  }
  if (element.matches("[data-pattern-resize]")) {
    if ((element as HTMLButtonElement).disabled) return;
    let nextColumns: number | undefined;
    if (event.key === "ArrowLeft" || event.key === "ArrowDown") nextColumns = patternColumns() - 1;
    if (event.key === "ArrowRight" || event.key === "ArrowUp") nextColumns = patternColumns() + 1;
    if (event.key === "Home") nextColumns = 1;
    if (event.key === "End") nextColumns = studioMode
      ? Math.min(state.perceptron.inputs, studioColumnsThatFit(classicNeuronFieldBounds().max, studioFieldScrollbar()))
      : state.perceptron.inputs;
    if (nextColumns !== undefined) {
      event.preventDefault();
      setPatternColumns(nextColumns);
    }
    return;
  }
  if (classicMode && element.matches(".classic-workspace-splitter")) {
    const bounds = classicNeuronFieldBounds();
    const increment = studioMode ? STUDIO_COLUMN_PITCH : event.shiftKey ? CLASSIC_NEURON_CARD_WIDTH : 12;
    let nextWidth: number | undefined;
    if (event.key === "ArrowLeft") nextWidth = classicNeuronFieldWidth - increment;
    if (event.key === "ArrowRight") nextWidth = classicNeuronFieldWidth + increment;
    if (event.key === "Home") nextWidth = bounds.min;
    if (event.key === "End") nextWidth = bounds.max;
    if (nextWidth !== undefined) {
      event.preventDefault();
      setClassicNeuronFieldWidth(nextWidth);
    }
    return;
  }
  const editing = element.matches("input, select, textarea, button, summary");
  if (event.code === "Space" && !editing && runtimeMode === "stopped" && !helpOpen) {
    event.preventDefault();
    doStep();
  }
});

window.addEventListener("beforeunload", stopAll);
render();
