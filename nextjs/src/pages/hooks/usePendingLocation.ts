import { create } from "zustand"

export type Location = {
  imageLocation: [number, number] | null
  mapLocation: [number, number] | null
  getPendingImageLocation: () => { x: number; y: number } | null
  getPendingMapLocation: () => { latitude: number; longitude: number } | null
  setPendingImageLocation: (location: [number, number]) => void
  setPendingMapLocation: (location: [number, number]) => void
}

export const usePendingLocation = create<Location>((set, get) => {
  return {
    imageLocation: null,
    mapLocation: null,

    getPendingImageLocation: () => {
      const location = get().imageLocation
      return location !== null ? { x: location[0], y: location[1] } : null
    },

    getPendingMapLocation: () => {
      const location = get().mapLocation
      return location !== null
        ? { latitude: location[0], longitude: location[1] }
        : null
    },

    setPendingImageLocation: (location: [number, number]) => {
      if (location) {
        set({ imageLocation: location })
      }
    },
    setPendingMapLocation: (location: [number, number]) => {
      if (location) {
        set({ mapLocation: location })
      }
    },
  }
})

export default usePendingLocation
