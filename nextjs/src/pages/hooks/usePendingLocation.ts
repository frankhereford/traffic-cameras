import { create } from "zustand"

export type Location = {
  x: number | null
  y: number | null
  latitude: number | null
  longitude: number | null
  getPendingImageLocation: () => (number | null)[]
  getPendingMapLocation: () => { x: number; y: number } | null
  setPendingImageLocation: (x: number, y: number) => void
  setPendingMapLocation: (location: number[]) => void
}

export const usePendingLocation = create<Location>((set, get) => {
  return {
    x: null,
    y: null,
    latitude: null,
    longitude: null,
    getPendingMapLocation: () => {
      const latitude = get().latitude
      const longitude = get().longitude
      return latitude !== null && longitude !== null
        ? { x: latitude, y: longitude }
        : null
    },
    getPendingMapLocation: () => [
      get().latitude ?? null,
      get().longitude ?? null,
    ],
    setPendingImageLocation: (x: number, y: number) => {
      if (x && y) {
        set({ x, y })
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
