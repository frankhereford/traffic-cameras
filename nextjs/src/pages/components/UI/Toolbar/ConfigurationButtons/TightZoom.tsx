import { useMapControls } from "~/pages/hooks/useMapControls"
import Button from "@mui/material/Button"

export default function TightZoom() {
  const zoomTight = useMapControls((state) => state.zoomTight)
  const setZoomTight = useMapControls((state) => state.setZoomTight)
  return (
    <Button
      className="mb-4 p-0"
      variant="contained"
      style={{ fontSize: "35px" }}
      onClick={() => setZoomTight(!zoomTight)}
    >
      {zoomTight ? "🚦" : "⛰️"}
    </Button>
  )
}
