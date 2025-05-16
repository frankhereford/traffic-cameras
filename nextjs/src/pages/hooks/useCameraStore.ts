import { create } from "zustand"

export type CameraStore = {
  camera: number | null
  previousCamera: number | null
  setCamera: (camera: number) => void
}

const CAMERA_STORAGE_KEY = "lastViewedCamera"

function getInitialCamera(): number | null {
  if (typeof window !== "undefined") {
    const stored = localStorage.getItem(CAMERA_STORAGE_KEY)
    if (stored !== null) {
      const parsed = parseInt(stored)
      if (!isNaN(parsed)) return parsed
    }
  }
  return null
}

export const useCameraStore = create<CameraStore>((set, get) => {
  return {
    camera: getInitialCamera(),
    previousCamera: null,
    setCamera: (cameraId) => {
      if (typeof window !== "undefined") {
        localStorage.setItem(CAMERA_STORAGE_KEY, String(cameraId))
      }
      set((state) => ({ camera: cameraId, previousCamera: state.camera }))
    },
  }
})

export default useCameraStore
