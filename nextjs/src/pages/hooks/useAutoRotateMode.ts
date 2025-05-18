import create from "zustand"
export type AutoRotateMode = {
  autoRotateMode: boolean
  setAutoRotateMode: (autoRotateMode: boolean) => void
}

export const useAutoRotateMode = create<AutoRotateMode>((set) => {
  return {
    autoRotateMode: false,
    setAutoRotateMode: (autoRotateMode: boolean) => {
      set({ autoRotateMode })
    },
  }
})

export default useAutoRotateMode
