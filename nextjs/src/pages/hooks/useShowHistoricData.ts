import create from "zustand"
export type ShowHistoricData = {
  showHistoricData: boolean
  setShowHistoricData: (showHistoricData: boolean) => void
}

export const useShowHistoricData = create<ShowHistoricData>((set) => {
  return {
    showHistoricData: false,
    setShowHistoricData: (showHistoricData: boolean) => {
      set({ showHistoricData })
    },
  }
})

export default useShowHistoricData
