import { create } from "zustand";
import type { Camera } from "~/pages/components/camerapicker";
import type { CameraPoint } from "~/pages/components/camera";

export type ApplicationState = {
  camera: number | null; // the camera ID defined by socrata data set
  setCamera: (camera: number) => void;

  cameraData: Camera | null; // the full record for the camera found in socrata data set
  setCameraData: (cameraData: Camera) => void;

  pendingCameraPoint: CameraPoint | null; // X,Y coordinates of the camera point being added
  setPendingCameraPoint: (cameraPoint: CameraPoint | null) => void;

  paneWidths: number[] | null; // the widths of the panes
  setPaneWidths: (paneWidths: number[]) => void;

  pendingMapPoint: google.maps.LatLng | null; // the lat/lng of the map point being added
  setPendingMapPoint: (mapPoint: google.maps.LatLng | null) => void;
};

// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
export const useApplicationStore = create<ApplicationState>((set, get) => {
  return {
    camera: null,
    setCamera: (camera: number) => {
      set({ camera });
    },

    cameraData: null,
    setCameraData: (cameraData: Camera | null) => {
      set({ cameraData });
    },

    pendingCameraPoint: null,
    setPendingCameraPoint: (pendingCameraPoint: CameraPoint | null) => {
      console.log("setPendingCameraPoint", pendingCameraPoint);
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
  };
});

export default useApplicationStore;
