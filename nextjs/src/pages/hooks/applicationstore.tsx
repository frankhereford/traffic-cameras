import { create } from "zustand";
import type { Camera } from "~/pages/components/camerapicker";
import type { CameraPoint } from "~/pages/components/camera";

export type ApplicationState = {
  reload: number;
  setReload: (reload: number) => void;

  camera: number | null; // the camera ID defined by socrata data set
  setCamera: (camera: number) => void;

  cameraData: Camera | null; // the full record for the camera found in socrata data set
  setCameraData: (cameraData: Camera) => void;

  allCameraData: Camera[] | null; // the full record for the camera found in socrata data set
  setAllCameraData: (camerasData: Camera[]) => void;

  pendingCameraPoint: CameraPoint | null; // X,Y coordinates of the camera point being added
  setPendingCameraPoint: (cameraPoint: CameraPoint | null) => void;

  paneWidths: number[] | null; // the widths of the panes
  setPaneWidths: (paneWidths: number[]) => void;

  pendingMapPoint: google.maps.LatLng | null; // the lat/lng of the map point being added
  setPendingMapPoint: (mapPoint: google.maps.LatLng | null) => void;

  mapZoom: number | null; // the zoom level of the map
  setMapZoom: (mapZoom: number) => void;
};

// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
export const useApplicationStore = create<ApplicationState>((set, get) => {
  return {
    reload: 0,
    setReload: (reload: number) => {
      set({ reload });
    },

    camera: null,
    setCamera: (camera: number) => {
      set({ camera });
    },

    cameraData: null,
    setCameraData: (cameraData: Camera | null) => {
      set({ cameraData });
    },

    allCameraData: null,
    setAllCameraData: (camerasData: Camera[]) => {
      set({ allCameraData: camerasData });
    },

    pendingCameraPoint: null,
    setPendingCameraPoint: (pendingCameraPoint: CameraPoint | null) => {
      set({ pendingCameraPoint });
    },

    paneWidths: null,
    setPaneWidths: (paneWidths: number[]) => {
      set({ paneWidths });
    },

    pendingMapPoint: null,
    setPendingMapPoint: (pendingMapPoint: google.maps.LatLng | null) => {
      set({ pendingMapPoint });
    },

    mapZoom: null,
    setMapZoom: (mapZoom: number) => {
      set({ mapZoom });
    },
  };
});

export default useApplicationStore;
