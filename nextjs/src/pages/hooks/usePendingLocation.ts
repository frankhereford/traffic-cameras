import { create } from "zustand"

export type Location = {
  x: number | null
  y: number | null
  latitude: number | null
  longitude: number | null
  getPendingImageLocation: () => (number | null)[]
  getPendingMapLocation: () => (number | null)[]
  setPendingImageLocation: (location: number[]) => void
  setPendingMapLocation: (location: number[]) => void
}

export const usePendingLocation = create<Location>((set, get) => {
  return {
    x: null,
    y: null,
    latitude: null,
    longitude: null,
    getPendingImageLocation: () => [get().x ?? null, get().y ?? null],
    getPendingMapLocation: () => [
      get().latitude ?? null,
      get().longitude ?? null,
    ],
    setPendingImageLocation: (location: number[]) => {
      if (location.length >= 2) {
        set({ x: location[0], y: location[1] })
      }
    },
    setPendingMapLocation: (location: number[]) => {
      if (location.length >= 2) {
        set({ latitude: location[0], longitude: location[1] })
      }
    },
  }
})

export default usePendingLocation
