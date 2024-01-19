import { create } from "zustand";

export type ApplicationState = {
  camera: number | null;
  setCamera: (camera: number) => void;
};

// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
export const useApplicationStore = create<ApplicationState>((set, get) => {
  return {
    camera: null,
    setCamera: (camera: number) => {
      set({ camera });
    },
  };
});

export default useApplicationStore;
