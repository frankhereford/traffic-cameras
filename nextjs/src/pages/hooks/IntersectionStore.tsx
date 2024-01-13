import { create } from "zustand"
import type { Camera } from "~/pages/components/cameraPicker"

export type Point = {
  x: number
  y: number
}

export type IntersectionState = {
  camera: number | null
  setCamera: (camera: number) => void
  cameraData: Camera | null
  setCameraData: (cameraData: Camera) => void
  cctvPendingPoint: Point | null
  setCctvPendingPoint: (cctvPendingPoint: Point) => void
  mapPendingPoint: number | null
  setMapPendingPoint: (mapPendingPoint: number) => void
}

export const useIntersectionStore = create<IntersectionState>((set) => ({
  camera: null,
  setCamera: (camera: number) => {
    set({ camera })
  },
  cameraData: null,
  setCameraData: (cameraData: Camera) => {
    set({ cameraData })
  },
  cctvPendingPoint: null,
  setCctvPendingPoint: (cctvPendingPoint: Point) => {
    console.log("setCctvPendingPoint", cctvPendingPoint)
    set({ cctvPendingPoint })
  },
  mapPendingPoint: null,
  setMapPendingPoint: (mapPendingPoint: number) => {
    set({ mapPendingPoint })
  },
}))

export default useIntersectionStore
