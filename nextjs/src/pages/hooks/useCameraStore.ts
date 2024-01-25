import { create } from "zustand"

export type CameraStore = {
  camera: number | null
  previousCamera: number | null
  setCamera: (camera: number) => void
}

export const useCameraStore = create<CameraStore>((set, get) => {
  return {
    camera: null,
    previousCamera: null,
    setCamera: (cameraId) =>
      set((state) => ({ camera: cameraId, previousCamera: state.camera })),
  }
})

export default useCameraStore
