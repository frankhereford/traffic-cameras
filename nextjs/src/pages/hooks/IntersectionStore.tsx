import { create } from "zustand"
import type { Camera } from "~/pages/components/cameraPicker"

export type Point = {
  x: number
  y: number
}

export type LatLng = {
  lat: number
  lng: number
}

export type CorrelatedPoint = {
  cctvPoint: Point
  mapPoint: LatLng
}

export type IntersectionState = {
  camera: number | null
  setCamera: (camera: number) => void
  cameraData: Camera | null
  setCameraData: (cameraData: Camera) => void
  cctvPendingPoint: Point | null
  setCctvPendingPoint: (cctvPendingPoint: Point) => void
  mapPendingPoint: LatLng | null
  setMapPendingPoint: (mapPendingPoint: LatLng) => void
  correlatedPoints: CorrelatedPoint[]
  setCorrelatedPoints: (correlatedPoints: CorrelatedPoint[]) => void
}

export const useIntersectionStore = create<IntersectionState>((set, get) => {
  return {
    camera: null,
    setCamera: (camera: number) => {
      set({ camera })
    },
    cameraData: null,
    setCameraData: (cameraData: Camera | null) => {
      set({ cameraData })
    },
    cctvPendingPoint: null,
    setCctvPendingPoint: (cctvPendingPoint: Point) => {
      // console.log("setCctvPendingPoint", cctvPendingPoint)
      set({ cctvPendingPoint })
    },
    mapPendingPoint: null,
    setMapPendingPoint: (mapPendingPoint: LatLng | null) => {
      // console.log("setMapPendingPoint", mapPendingPoint)
      set({ mapPendingPoint })
    },
    correlatedPoints: [],
    setCorrelatedPoints: (correlatedPoints: CorrelatedPoint[]) => {
      // console.log("setCorrelatedPoints", correlatedPoints)
      set({ correlatedPoints })
    },
  }
})

export default useIntersectionStore
