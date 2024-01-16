/* eslint-disable @typescript-eslint/no-unsafe-assignment */
import { create } from "zustand"
import type { Camera } from "~/pages/components/cameraPicker"

export type WarpedLabel = {
  lat: number
  lng: number
}

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

type BoundingBox = {
  Width: number
  Height: number
  Left: number
  Top: number
}

type Instance = {
  BoundingBox: BoundingBox
  Confidence: number
}

type Category = {
  Name: string
}

type Parent = {
  Name: string
}

type Alias = {
  Name: string
}

type Label = {
  Name: string
  Confidence: number
  Instances: Instance[]
  Parents: Parent[]
  Aliases: Alias[]
  Categories: Category[]
}

type ResponseMetadata = {
  RequestId: string
  HTTPStatusCode: number
  HTTPHeaders: {
    "x-amzn-requestid": string
    "content-type": string
    "content-length": string
    date: string
  }
  RetryAttempts: number
}

type ImageRecognitionResponse = {
  Labels: Label[]
  LabelModelVersion: string
  ResponseMetadata: ResponseMetadata
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
  cctvImage: string | null
  setCctvImage: (cctvImage: string | null) => void
  recognition: ImageRecognitionResponse | null
  setRecognition: (response: ImageRecognitionResponse | null) => void
  warpedLabels: WarpedLabel[] | null
  setWarpedLabels: (warpedLabels: WarpedLabel[] | null) => void
  mapPeekPoint: LatLng | null
  setMapPeekPoint: (mapPeekPoint: LatLng) => void
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
      console.log("setCctvPendingPoint", cctvPendingPoint)
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
    cctvImage: null,
    setCctvImage: (cctvImage: string | null) => {
      set({ cctvImage })
    },
    recognition: null,
    setRecognition: (response: ImageRecognitionResponse | null) => {
      set({ recognition: response })
    },
    warpedLabels: null,
    setWarpedLabels: (warpedLabels: WarpedLabel[] | null) => {
      set({ warpedLabels })
    },
    mapPeekPoint: null,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    setMapPeekPoint: (mapPeekPoint: any) => {
      set({ mapPeekPoint })
    },
  }
})

export default useIntersectionStore
