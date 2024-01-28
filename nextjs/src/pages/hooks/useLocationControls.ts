import { create } from "zustand"

export type LocationsControls = {
  showLocations: boolean | null
  setShowLocations: (showLocations: boolean) => void
}

export const useLocationControls = create<LocationsControls>(
  (set: (partial: Partial<LocationsControls>) => void) => {
    return {
      showLocations: false,
      setShowLocations: (showLocations: boolean) => {
        set({ showLocations })
      },
    }
  },
)

export default useLocationControls
