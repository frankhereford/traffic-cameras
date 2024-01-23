import { create } from "zustand"

export type CameraStore = {
  camera: number | null
  setCamera: (camera: number) => void
}

export const useCameraStore = create<CameraStore>((set, get) => {
  return {
    camera: null,
    setCamera: (camera: number) => {
      set({ camera })
    },
  }
})

export default useCameraStore
