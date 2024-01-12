import { create } from "zustand"

type IntersectionState = {
  camera: number | null
}

const useIntersectionStore = create<IntersectionState>((set) => ({
  camera: null, // 533
  setCamera: (camera: number) => {
    console.log(`New camera value: ${camera}`)
    set({ camera })
  },
}))

export default useIntersectionStore
