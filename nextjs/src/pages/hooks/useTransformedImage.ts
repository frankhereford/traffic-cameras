import create from "zustand"
export type ShowTransformedImage = {
  showTransformedImage: boolean
  setShowTransformedImage: (showHistoricData: boolean) => void
}

export const useShowTransformedImage = create<ShowTransformedImage>((set) => {
  return {
    showTransformedImage: false,
    setShowTransformedImage: (showTransformedImage: boolean) => {
      set({ showTransformedImage })
    },
  }
})

export default useShowTransformedImage
