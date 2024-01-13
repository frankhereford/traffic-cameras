import { create } from "zustand"
import type { Camera } from "~/pages/components/cameraPicker"

export type IntersectionState = {
  camera: number | null
  setCamera: (camera: number) => void
  cameraData: Camera | null
  setCameraData: (cameraData: Camera) => void
}

export const useIntersectionStore = create<IntersectionState>((set) => ({
  camera: null,
  setCamera: (camera: number) => {
    console.log(`New camera value: ${camera}`)
    set({ camera })
  },
  cameraData: null,
  setCameraData: (cameraData: Camera) => {
    console.log(`New cameraData value: ${JSON.stringify(cameraData)}`)
    set({ cameraData })
  },
}))

export default useIntersectionStore
