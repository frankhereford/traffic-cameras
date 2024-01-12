import { create } from "zustand"

type Coordinate = {
  cctv: [number, number]
  map: [number, number]
}

type CoordinateArray = Coordinate[]

interface CoordinateState {
  coordinates: CoordinateArray
  mapCenter: [number, number]
  addCoordinates: (
    cctvCoords: [number, number],
    mapCoords: [number, number],
  ) => void
  setMapCenter: (mapCenter: [number, number]) => void
}

const useCoordinateStore = create<CoordinateState>((set) => ({
  coordinates: [],
  mapCenter: [0, 0],

  addCoordinates: (cctvCoords: [number, number], mapCoords: [number, number]) =>
    set((state) => ({
      coordinates: [...state.coordinates, { cctv: cctvCoords, map: mapCoords }],
    })),

  setMapCenter: (mapCenter: [number, number]) =>
    set((state) => ({ mapCenter: mapCenter })),
}))

export default useCoordinateStore
