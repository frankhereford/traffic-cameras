import { create } from "zustand"

export type AutocompleteFocus = {
  isFocus: boolean | null
  setIsFocus: (zoomTight: boolean) => void
}

export const useAutocompleteFocus = create<AutocompleteFocus>((set) => {
  return {
    isFocus: false,
    setIsFocus: (isFocus: boolean) => {
      set({ isFocus })
    },
  }
})

export default useAutocompleteFocus
