import { create } from "zustand";
// We expect @react-google-maps/api to provide the google.maps types globally
// or we might need to import LatLngBounds directly from it if available.

interface MapViewportState {
  bounds: google.maps.LatLngBounds | null;
  setMapBounds: (bounds: google.maps.LatLngBounds | null) => void; // Renamed for clarity from setBounds
}

const useMapViewportStore = create<MapViewportState>((set) => ({
  bounds: null,
  setMapBounds: (bounds) => set({ bounds }),
}));

export default useMapViewportStore; 
