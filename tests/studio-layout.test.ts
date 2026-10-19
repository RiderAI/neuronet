import { describe, expect, it } from "vitest";
import {
  configurePerceptron, createNetwork, deserializeState, resizeNetwork,
  runStep, serializeState, switchModel, trainStep, validateState,
} from "../src/engine";
import {
  studioColumns, studioColumnsThatFit, studioPaneWidth, studioPatternColumns,
} from "../src/studio-layout";

describe("Studio: геометрия общей сетки", () => {
  it.each([[1, 88], [3, 244], [4, 322], [5, 400], [32, 2506]])(
    "%i колонок занимают %i пикселей без полосы прокрутки",
    (columns, width) => {
      expect(studioPaneWidth(columns)).toBe(width);
      expect(studioColumnsThatFit(width)).toBe(columns);
      expect(studioColumnsThatFit(width - 1)).toBe(Math.max(1, columns - 1));
    },
  );

  it("компенсирует реальную полосу прокрутки, сохраняя число колонок", () => {
    for (const columns of [1, 3, 4, 5, 32]) {
      for (const gutter of [0, 8, 15, 17]) {
        expect(studioPaneWidth(columns, gutter)).toBe(studioPaneWidth(columns) + gutter);
        expect(studioColumnsThatFit(studioPaneWidth(columns, gutter), gutter)).toBe(columns);
      }
    }
    expect(studioColumnsThatFit(322, 15)).toBe(3);
    expect(studioColumnsThatFit(337, 15)).toBe(4);
  });

  it("ограничивает видимую сетку доступной шириной и пределом 32", () => {
    expect(studioColumnsThatFit(390)).toBe(4);
    expect(studioColumnsThatFit(320)).toBe(3);
    expect(studioColumnsThatFit(800)).toBe(10);
    expect(studioColumnsThatFit(0)).toBe(1);
    expect(studioColumnsThatFit(10000)).toBe(32);
    expect(studioColumns(-9)).toBe(1);
    expect(studioColumns(99)).toBe(32);
    expect(studioColumns(4.6)).toBe(5);
    expect(studioColumns(Number.NaN)).toBe(4);
  });

  it("ограничивает только редактор образа при одном-двух входах", () => {
    for (const preferred of [4, 8, 32]) {
      expect(studioPatternColumns(preferred, 1)).toBe(1);
      expect(studioPatternColumns(preferred, 2)).toBe(2);
      expect(studioColumns(preferred)).toBe(preferred);
    }
    expect(studioPatternColumns(5, 25)).toBe(5);
    expect(studioPatternColumns(3, 25)).toBe(3);
  });
});

describe("Studio: необязательная раскладка проекта", () => {
  it("принимает старый проект без метаданных и новые значения 1..32", () => {
    const source = createNetwork();
    expect(validateState(source)).toBe(true);
    expect(deserializeState(source).state.layout).toBeUndefined();
    for (const neuronColumns of [1, 4, 32]) {
      const candidate = { ...source, layout: { neuronColumns } };
      expect(validateState(candidate)).toBe(true);
      expect(deserializeState(candidate).state.layout).toEqual({ neuronColumns });
    }
  });

  it.each([null, [], {}, { neuronColumns: 0 }, { neuronColumns: 33 },
    { neuronColumns: 2.5 }, { neuronColumns: "4" }, { neuronColumns: Number.NaN }])(
    "отклоняет неверную форму метаданных: %j", (layout) => {
      const candidate = { ...createNetwork(), layout };
      expect(validateState(candidate)).toBe(false);
      expect(() => deserializeState(candidate)).toThrow();
    },
  );

  it("хранит широкое поле отдельно от допустимых колонок двух входов", () => {
    const source = configurePerceptron(createNetwork(), 2, 1);
    source.layout = { neuronColumns: 32 };
    source.perceptron.patternColumns = 2;
    expect(validateState(source)).toBe(true);
    expect(deserializeState(JSON.parse(serializeState(source))).state.layout).toEqual({ neuronColumns: 32 });
    source.perceptron.patternColumns = 3;
    expect(validateState(source)).toBe(false);
  });

  it("сохраняет раскладку 5×5 вместе с весами, входами, целями и набором", () => {
    const source = configurePerceptron(createNetwork(), 25, 2);
    source.layout = { neuronColumns: 5 };
    source.perceptron.patternColumns = 5;
    source.perceptron.desired = [0, 1];
    source.perceptron.trainingExamples = [{ name: "Образ", inputs: Array.from({ length: 25 }, (_, i) => i % 2), desired: [1, 0] }];
    source.neurons[3]!.constantInput = 1;
    source.weights[3]![25] = 0.125;
    source.weights[8]![26] = -0.75;
    const before = structuredClone(source);
    const restored = deserializeState(JSON.parse(serializeState(source))).state;
    expect(restored).toEqual(before);
    expect(source).toEqual(before);
  });

  it("раскладка не меняет расчёт шага и коррекцию весов", () => {
    const source = configurePerceptron(createNetwork(), 2, 2);
    source.neurons[0]!.constantInput = 1;
    source.neurons[1]!.constantInput = 0.5;
    source.weights[0]![2] = 0.2;
    source.perceptron.desired = [0, 1];
    const styled = structuredClone(source);
    styled.layout = { neuronColumns: 8 };
    for (const operation of [runStep, trainStep]) {
      const plainResult = operation(source);
      const styledResult = operation(styled);
      expect(styledResult.state.layout).toEqual({ neuronColumns: 8 });
      delete styledResult.state.layout;
      expect(styledResult).toEqual(plainResult);
    }
    expect(styled.layout).toEqual({ neuronColumns: 8 });
  });

  it("сохраняет предпочтение при изменении модели и количества нейронов", () => {
    const source = createNetwork();
    source.layout = { neuronColumns: 5 };
    const variants = [resizeNetwork(source, 32), resizeNetwork(source, 2),
      configurePerceptron(source, 1, 1), switchModel(source, "perceptron")];
    for (const variant of variants) {
      expect(variant.layout).toEqual({ neuronColumns: 5 });
      expect(validateState(variant)).toBe(true);
    }
    expect(switchModel(variants[3]!, "scatter").layout).toEqual({ neuronColumns: 5 });
  });
});
