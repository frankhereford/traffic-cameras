import { create } from "zustand";
import type { Camera } from "~/pages/components/camerapicker";

export type ApplicationState = {
  camera: number | null;
  setCamera: (camera: number) => void;
  cameraData: Camera | null;
  setCameraData: (cameraData: Camera) => void;
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
  };
});

export default useApplicationStore;
