import { create } from "zustand"

export type Location = {
  imageLocation: [number, number] | null
  mapLocation: [number, number] | null
  getPendingImageLocation: () => { x: number; y: number } | null
  getPendingMapLocation: () => { latitude: number; longitude: number } | null
  getCorrelatedLocation: () => { x: number; y: number } | null
  setPendingImageLocation: (location: { x: number; y: number }) => void
  setPendingMapLocation: (location: {
    latitude: number
    longitude: number
  }) => void
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

    getCorrelatedLocation: () => {
      const imageLocation = get().imageLocation
      const mapLocation = get().mapLocation
      if (imageLocation !== null && mapLocation !== null) {
        return {
          x: imageLocation[0],
          y: imageLocation[1],
          latitude: mapLocation[0],
          longitude: mapLocation[1],
        }
      }
      return null
    },

    setPendingImageLocation: (location: { x: number; y: number }) => {
      if (location) {
        set({ imageLocation: [location.x, location.y] })
      }
    },
    setPendingMapLocation: (
      location: {
        latitude: number
        longitude: number
      } | null,
    ) => {
      if (location) {
        set({ mapLocation: [location.latitude, location.longitude] })
      }
    },
  }
})

export default usePendingLocation
