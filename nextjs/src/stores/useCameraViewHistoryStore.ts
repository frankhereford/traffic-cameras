import { create } from "zustand";

interface CameraViewHistoryState {
  viewedCameraIds: number[]; // Ordered list, newest at the end
  addCameraToViewHistory: (coaId: number) => void;
}

const MAX_HISTORY_LENGTH = 200; // Optional: Limit history size

const useCameraViewHistoryStore = create<CameraViewHistoryState>((set, get) => ({
  viewedCameraIds: [],
  addCameraToViewHistory: (coaId: number) => {
    set(state => {
      const currentHistory = state.viewedCameraIds.filter(id => id !== coaId);
      const newHistory = [...currentHistory, coaId];
      // Optional: Trim history if it exceeds max length
      if (newHistory.length > MAX_HISTORY_LENGTH) {
        return { viewedCameraIds: newHistory.slice(newHistory.length - MAX_HISTORY_LENGTH) };
      }
      return { viewedCameraIds: newHistory };
    });
  },
}));

export default useCameraViewHistoryStore; 