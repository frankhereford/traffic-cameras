import create from "zustand"

export type ShowTransformedImage = {
  showTransformedImage: boolean
  setShowTransformedImage: (showHistoricData: boolean) => void
  opacity: number
  setOpacity: (opacity: number) => void
}

export const useShowTransformedImage = create<ShowTransformedImage>((set) => {
  return {
    showTransformedImage: false,
    setShowTransformedImage: (showTransformedImage: boolean) => {
      set({ showTransformedImage })
    },
    opacity: 1,
    setOpacity: (opacity: number) => {
      set({ opacity })
    },
  }
})

export default useShowTransformedImage
