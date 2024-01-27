import { useEffect, useState } from "react"
import { useMapControls } from "~/pages/hooks/useMapControls"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"

export default function TightZoom() {
  const zoomTight = useMapControls((state) => state.zoomTight)
  const setZoomTight = useMapControls((state) => state.setZoomTight)
  const [isHovered, setIsHovered] = useState(false)

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "w") {
        setZoomTight(!zoomTight)
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [zoomTight])

  return (
    <Tooltip title="Toggle zoom">
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px", position: "relative" }}
        onClick={() => setZoomTight(!zoomTight)}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {zoomTight ? "🚦" : "⛰️"}
        {isHovered && (
          <span
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              fontSize: "50px",
              opacity: 0.15,
            }}
          >
            w
          </span>
        )}
      </Button>
    </Tooltip>
  )
}
