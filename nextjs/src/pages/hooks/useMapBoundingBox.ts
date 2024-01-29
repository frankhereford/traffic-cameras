import { create } from "zustand"

export type BoundingBox = {
  xMin: number | null
  xMax: number | null
  yMin: number | null
  yMax: number | null
  setBoundingBox: (
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
  ) => void
}

export const useBoundingBox = create<BoundingBox>((set) => {
  return {
    xMin: null,
    xMax: null,
    yMin: null,
    yMax: null,
    setBoundingBox: (xMin, xMax, yMin, yMax) =>
      set(() => ({ xMin, xMax, yMin, yMax })),
  }
})

export default useBoundingBox
