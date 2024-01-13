import { create } from "zustand"

export type IntersectionState = {
  camera: number | null
  setCamera: (camera: number) => void
}

export const useIntersectionStore = create<IntersectionState>((set) => ({
  camera: null,
  setCamera: (camera: number) => {
    console.log(`New camera value: ${camera}`)
    set({ camera })
  },
}))

export default useIntersectionStore
