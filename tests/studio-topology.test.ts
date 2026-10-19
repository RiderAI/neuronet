import { describe, expect, it } from "vitest";
import { configurePerceptron, createNetwork } from "../src/engine";
import { renderStudioTopology, renderStudioTopologyPlot, studioTopologyGeometry } from "../src/studio-topology";

describe("Studio direct topology drawing", () => {
  it.each([[1, 31], [31, 1], [16, 16], [14, 1], [25, 2], [2, 2]])("draws every node in a %i→%i network inside narrow and wide viewports", (inputs, outputs) => {
    for (const width of [180, 220, 320, 500]) {
      const geometry = studioTopologyGeometry(inputs, outputs, width);
      expect(geometry.inputs).toHaveLength(inputs);
      expect(geometry.outputs).toHaveLength(outputs);
      for (const node of [...geometry.inputs, ...geometry.outputs]) {
        expect(node.x - 9).toBeGreaterThanOrEqual(0);
        expect(node.x + 9).toBeLessThanOrEqual(width);
        expect(node.y - 9).toBeGreaterThanOrEqual(0);
        expect(node.y + 9).toBeLessThanOrEqual(geometry.height);
      }
      expect(Math.max(...geometry.inputs.map(node => node.x))).toBeLessThan(Math.min(...geometry.outputs.map(node => node.x)));
    }
  });

  it("has only input and output layers; rendering does not change the model", () => {
    const state = configurePerceptron(createNetwork(), 14, 2);
    state.weights[0]![14] = 0.25;
    state.neurons[0]!.output = 1;
    const before = structuredClone(state);
    const html = renderStudioTopology(state);
    expect(html.match(/data-topology-neuron=/g)).toHaveLength(16);
    expect(html.match(/data-topology-source=/g)).toHaveLength(28);
    expect(html).toContain('class="studio-edge connected signal"');
    expect(html).toContain("w = 0,25");
    expect(html).toContain("H15–H16");
    expect(state).toEqual(before);
  });

  it("shows the current values and the actual connection weights, including zero", () => {
    const state = configurePerceptron(createNetwork(), 1, 1);
    state.neurons[0]!.potential = 0.4;
    state.neurons[0]!.output = 0.4;
    state.neurons[1]!.potential = -0.2;
    state.weights[0]![1] = -0.5;
    const html = renderStudioTopologyPlot(state, 250);
    expect(html).toContain("P = 0,4, z = 0,4");
    expect(html).toContain("P = -0,2, z = 0");
    expect(html).toContain("w = -0,5");
    expect(html).toContain('viewBox="0 0 250 80"');
  });
});
