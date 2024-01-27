import { create } from "zustand"

export type Location = {
  x: number | null
  y: number | null
  latitude: number | null
  longitude: number | null
  getPendingImageLocation: () => { x: number; y: number } | null
  getPendingMapLocation: () => { latitude: number; longitude: number } | null
  setPendingImageLocation: (x: number, y: number) => void
  setPendingMapLocation: (latitude: number, longitude: number) => void
}

export const usePendingLocation = create<Location>((set, get) => {
  return {
    x: null,
    y: null,
    latitude: null,
    longitude: null,

    getPendingImageLocation: () => {
      const x = get().x
      const y = get().y
      return x !== null && y !== null ? { x, y } : null
    },

    getPendingMapLocation: () => {
      const latitude = get().latitude
      const longitude = get().longitude
      return latitude !== null && longitude !== null
        ? { latitude, longitude }
        : null
    },

    setPendingImageLocation: (x: number, y: number) => {
      if (x && y) {
        set({ x, y })
      }
    },
    setPendingMapLocation: (latitude: number, longitude: number) => {
      if (latitude && longitude) {
        set({ latitude, longitude })
      }
    },
  }
})

export default usePendingLocation
