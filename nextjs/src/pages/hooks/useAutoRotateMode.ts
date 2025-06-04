import create from "zustand"

export const MIN_INTERVAL_S = 3
export const MAX_INTERVAL_S = 15
export const DEFAULT_INTERVAL_S = 15

export type AutoRotateModeState = { // Renamed for clarity
  autoRotateMode: boolean
  setAutoRotateMode: (autoRotateMode: boolean) => void
  autoRotateIntervalS: number
  setAutoRotateIntervalS: (interval: number) => void
}

export const useAutoRotateMode = create<AutoRotateModeState>((set) => {
  return {
    autoRotateMode: false,
    setAutoRotateMode: (autoRotateMode: boolean) => {
      set({ autoRotateMode })
    },
    autoRotateIntervalS: DEFAULT_INTERVAL_S,
    setAutoRotateIntervalS: (interval: number) => {
      set({ autoRotateIntervalS: interval })
    },
  }
})

export default useAutoRotateMode
