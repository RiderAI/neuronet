import { describe, expect, it } from "vitest";
import { applyActivation, createNetwork } from "../src/engine";
import { activationPlotGeometry, renderActivationPlot } from "../src/activation-plot";
import type { ActivationKind } from "../src/types";

describe("activation preview matches the engine", () => {
  it.each<ActivationKind>(["hard-limit", "linear"])("samples gated %s using the engine, on the actual P axis", kind => {
    for (const threshold of [-2, -.2, 0, .4, 2, 10]) {
      const neuron = { ...createNetwork().neurons[0]!, activation: kind, threshold, linear: { slope: 1.5, intercept: .2 } };
      const before = structuredClone(neuron);
      const g = activationPlotGeometry(neuron);
      expect(g.inactive.at(-1)).toEqual({ potential: threshold, output: 0 });
      expect(g.active[0]!.potential).toBe(threshold);
      for (const point of g.active.slice(1)) {
        expect(point.potential).toBeGreaterThan(threshold);
        expect(point.output).toBe(applyActivation(kind, point.potential, threshold, neuron.linear));
      }
      expect(neuron).toEqual(before);
    }
  });

  it.each([-5, -.4, 0, .4, 5])("draws a continuous sigmoid across Q=%s with centre at .5", threshold => {
    const neuron = { ...createNetwork().neurons[0]!, activation: "sigmoid" as const, threshold };
    const g = activationPlotGeometry(neuron);
    expect(g.rightLimit).toBe(.5);
    expect(g.discontinuous).toBe(false);
    expect(g.continuous).toBe(true);
    expect(g.inactive).toEqual([]);
    expect(g.active[96]).toEqual({ potential: threshold, output: .5 });
    expect(g.active[0]!.output).toBeLessThan(.02);
    expect(g.active.at(-1)!.output).toBeGreaterThan(.98);
    for (const point of g.active) {
      expect(point.output).toBe(applyActivation("sigmoid", point.potential, threshold));
      expect(point.output).toBeGreaterThan(0);
      expect(point.output).toBeLessThan(1);
    }
    const svg = renderActivationPlot(neuron);
    expect(svg).not.toContain('data-activation-part="inactive"');
    expect(svg.match(/data-activation-part="curve"/g)).toHaveLength(1);
    expect(svg).toContain('data-activation-boundary="centre"');
    expect(svg).not.toContain('data-activation-boundary="open"');
    expect(svg).toContain(`data-threshold="${threshold}"`);
    expect(svg).toContain("при P = Q выход равен 0,5");
    expect(svg).not.toContain("При P ≤ Q выход равен 0");
    expect(svg).not.toMatch(/NaN|Infinity/);
  });

  it("keeps Q in the input coordinate instead of cancelling it", () => {
    const neuron = { ...createNetwork().neurons[0]!, activation: "sigmoid" as const };
    const paths = [0, .4].map(threshold => renderActivationPlot({ ...neuron, threshold }).match(/data-activation-part="curve" d="([^"]+)"/)![1]);
    expect(paths[0]).not.toEqual(paths[1]);
  });

  it("shows the hard-limit jump with open1 and closed0 boundary points", () => {
    const g = activationPlotGeometry(createNetwork().neurons[0]!);
    expect(g.rightLimit).toBe(1);
    expect(g.active.every(p => p.output === 1)).toBe(true);
    expect(g.inactive.at(-1)!.output).toBe(0);
  });

  it.each([0, .25, -.5])("handles linear intercept %s without silently clipping its graph", intercept => {
    const neuron = { ...createNetwork().neurons[0]!, activation: "linear" as const, linear: { slope: -2, intercept } };
    const g = activationPlotGeometry(neuron);
    expect(g.rightLimit).toBe(intercept);
    expect(g.discontinuous).toBe(intercept !== 0);
    expect(g.yMin).toBeLessThan(Math.min(...g.active.map(p => p.output)));
    expect(g.yMax).toBeGreaterThan(Math.max(...g.active.map(p => p.output)));
    const svg = renderActivationPlot(neuron);
    expect(svg.includes('data-activation-boundary="open"')).toBe(intercept !== 0);
  });
});
