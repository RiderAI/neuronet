import { applyActivation, finite } from "./engine";
import type { Neuron } from "./types";

type PreviewNeuron = Pick<Neuron, "id" | "activation" | "threshold" | "linear">;
export interface ActivationPlotPoint { potential: number; output: number }

/** Sigmoid is continuous. Separate branches preserve the strict P > Q gate
 * of hard-limit and linear functions, including their excluded right limit. */
export function activationPlotGeometry(neuron: PreviewNeuron) {
  const threshold = finite(neuron.threshold);
  const continuous = neuron.activation === "sigmoid";
  const xMin = Math.min(continuous ? -4 : -2, threshold - (continuous ? 4 : 2));
  const xMax = Math.max(4, threshold + 4);
  const rightLimit = neuron.activation === "sigmoid" ? 0.5
    : neuron.activation === "hard-limit" ? 1 : finite(neuron.linear.intercept);
  const inactive: ActivationPlotPoint[] = continuous ? [] : [
    { potential: xMin, output: 0 }, { potential: threshold, output: 0 },
  ];
  const active: ActivationPlotPoint[] = continuous ? Array.from({ length: 193 }, (_, index) => {
    // Include the centre exactly, even when Q is far from the origin.
    const potential = index === 96 ? threshold : index < 96
      ? xMin + (index / 96) * (threshold - xMin)
      : threshold + ((index - 96) / 96) * (xMax - threshold);
    return { potential, output: applyActivation(neuron.activation, potential, threshold, neuron.linear) };
  }) : [
    { potential: threshold, output: rightLimit },
    ...Array.from({ length: 96 }, (_, index) => {
      const potential = threshold + ((index + 1) / 96) * (xMax - threshold);
      return { potential, output: applyActivation(neuron.activation, potential, threshold, neuron.linear) };
    }),
  ];
  const outputs = active.map(point => finite(point.output));
  const outputMin = Math.min(0, ...outputs);
  const outputMax = Math.max(1, ...outputs);
  const padding = (outputMax - outputMin) * 0.12;
  return {
    threshold, xMin, xMax, inactive, active, rightLimit, continuous,
    discontinuous: !continuous && rightLimit !== 0,
    yMin: outputMin - padding, yMax: outputMax + padding,
    outputMin, outputMax,
  };
}

function label(value: number): string {
  if (value !== 0 && (Math.abs(value) >= 1000 || Math.abs(value) < .001)) return value.toExponential(1).replace(".", ",");
  return Number(value.toFixed(2)).toString().replace(".", ",");
}

export function renderActivationPlot(neuron: PreviewNeuron): string {
  const g = activationPlotGeometry(neuron);
  const left = 26, right = 141, top = 18, bottom = 124;
  const x = (potential: number) => left + (potential - g.xMin) / (g.xMax - g.xMin) * (right - left);
  const y = (output: number) => bottom - (finite(output) - g.yMin) / (g.yMax - g.yMin) * (bottom - top);
  const path = (points: ActivationPlotPoint[]) => points.map((point, i) => `${i ? "L" : "M"}${x(point.potential).toFixed(2)} ${y(point.output).toFixed(2)}`).join(" ");
  const thresholdX = x(g.threshold).toFixed(2);
  const zeroY = y(0).toFixed(2);
  const ticks = [...new Set(neuron.activation === "linear" ? [g.outputMin, 0, g.outputMax] : [0, .5, 1])];
  const color = "var(--skin-accent, #f00)";
  const description = `Вход P, выход z. ${g.continuous ? "Сигмоидальная функция z = 1/(1+exp(−(P−Q))) при любом P; при P = Q выход равен 0,5." : "При P ≤ Q выход равен 0."} Порог Q = ${label(g.threshold)}.${g.discontinuous ? " Закрашенная точка входит в график; незакрашенная обозначает предел справа и не входит." : ""}`;
  return `<svg viewBox="0 0 150 150" role="img" data-threshold="${g.threshold}" aria-label="График функции активации нейрона ${neuron.id}: z от P">
    <title>${description}</title>
    <rect x="0.5" y="0.5" width="149" height="149" fill="#fff" stroke="#b8b8b8" />
    <g fill="#526775" font-family="Segoe UI, sans-serif" font-size="8">
      ${ticks.map(value => `<path d="M${left} ${y(value).toFixed(2)}H${right}" fill="none" stroke="#e4ebef"/><text x="22" y="${(y(value) + 2.5).toFixed(2)}" text-anchor="end">${label(value)}</text>`).join("")}
      <path d="M${left} ${top}V${bottom} M${left} ${zeroY}H${right}" fill="none" stroke="#657b88" stroke-width=".8"/>
      <text x="9" y="12" font-size="10">z</text><text x="143" y="${(y(0) - 5).toFixed(2)}" text-anchor="end" font-size="10">P</text>
      <text x="${left}" y="146" text-anchor="middle">${label(g.xMin)}</text><text x="${right}" y="146" text-anchor="end">${label(g.xMax)}</text>
      <path data-activation-part="threshold" d="M${thresholdX} ${top}V${bottom}" fill="none" stroke="#9aadb7" stroke-dasharray="2 3" stroke-width=".8"/>
      <text x="${thresholdX}" y="137" text-anchor="middle">Q=${label(g.threshold)}</text>
    </g>
    ${g.continuous ? "" : `<path data-activation-part="inactive" d="${path(g.inactive)}" fill="none" stroke="${color}" stroke-width="1.5"/>`}
    <path data-activation-part="${g.continuous ? "curve" : "active"}" d="${path(g.active)}" fill="none" stroke="${color}" stroke-width="1.5"/>
    <circle data-activation-boundary="${g.continuous ? "centre" : "closed"}" cx="${thresholdX}" cy="${g.continuous ? y(.5).toFixed(2) : zeroY}" r="2.3" fill="${color}"/>
    ${g.discontinuous ? `<circle data-activation-boundary="open" cx="${thresholdX}" cy="${y(g.rightLimit).toFixed(2)}" r="2.6" fill="#fff" stroke="${color}" stroke-width="1.2"/>` : ""}
  </svg>`;
}
