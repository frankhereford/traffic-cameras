import { create } from "zustand"

export type MapControls = {
  zoomTight: boolean | null
  setZoomTight: (zoomTight: boolean) => void
}

export const useMapControls = create<MapControls>((set, get) => {
  return {
    zoomTight: false,
    setZoomTight: (zoomTight: boolean) => {
      set({ zoomTight })
    },
  }
})

export default useMapControls
