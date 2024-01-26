import { create } from "zustand"

export type LocationStore = {
  x: number | null
  y: number | null
  latitude: number | null
  longitude: number | null
  setImageLocation: (location: number[]) => void
}

export const useCameraStore = create<LocationStore>((set, get) => {
  return {
    x: null,
    y: null,
    latitude: null,
    longitude: null,
    setImageLocation: (location: number[]) =>
      set({ x: location[0], y: location[1] }),
    setMapLocation: (location: number[]) =>
      set({ latitude: location[0], longitude: location[1] }),
  }
})

export default useCameraStore
