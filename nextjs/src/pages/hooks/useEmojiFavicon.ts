import { create } from "zustand"

export type EmojiFaviconState = {
  emoji: string
  setEmoji: (emoji: string) => void
}

export const useEmojiFavicon = create<EmojiFaviconState>((set) => ({
  emoji: "🚦",
  setEmoji: (emoji: string) => set({ emoji }),
}))

export default useEmojiFavicon
