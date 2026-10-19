/** Studio grid geometry. Kept independent from all network calculations. */
export const STUDIO_CARD_WIDTH = 72;
export const STUDIO_GRID_GAP = 6;
export const STUDIO_GRID_PADDING = 16;
export const STUDIO_COLUMN_PITCH = STUDIO_CARD_WIDTH + STUDIO_GRID_GAP;
export const STUDIO_SPLITTER_WIDTH = 10;

export function studioColumns(value: number, maximum = 32): number {
  return Math.max(1, Math.min(maximum, Math.round(Number.isFinite(value) ? value : 4)));
}

export function studioPaneWidth(columns: number, scrollbar = 0): number {
  return STUDIO_GRID_PADDING + studioColumns(columns) * STUDIO_COLUMN_PITCH - STUDIO_GRID_GAP + Math.max(0, scrollbar);
}

export function studioColumnsThatFit(width: number, scrollbar = 0): number {
  const available = Math.max(0, width - Math.max(0, scrollbar) - STUDIO_GRID_PADDING + STUDIO_GRID_GAP);
  return Math.max(1, Math.min(32, Math.floor((available + 0.01) / STUDIO_COLUMN_PITCH)));
}

export function studioPatternColumns(fieldColumns: number, inputs: number): number {
  return Math.min(studioColumns(fieldColumns), Math.max(1, inputs));
}
