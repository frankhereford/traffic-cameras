import { create } from 'zustand'

type Coordinate = {
    cctv: [number, number];
    map: [number, number];
};

type CoordinateArray = Coordinate[];

interface CoordinateState {
    coordinates: CoordinateArray;
    addCoordinates: (cctvCoords: [number, number], mapCoords: [number, number]) => void;
}

const useCoordinateStore = create<CoordinateState>((set) => ({
    coordinates: [],
    addCoordinates: (cctvCoords: [number, number], mapCoords: [number, number]) =>
        set((state) => ({
            coordinates: [...state.coordinates, { cctv: cctvCoords, map: mapCoords }],
        })),
}))

export default useCoordinateStore;