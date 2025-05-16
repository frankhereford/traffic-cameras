import { useEffect, useState } from "react"
import { useMapControls } from "~/pages/hooks/useMapControls"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import useEmojiFavicon from "~/pages/hooks/useEmojiFavicon"

export default function ToggleMapFollow() {
  const zoomTight = useMapControls((state) => state.zoomTight)
  const setZoomTight = useMapControls((state) => state.setZoomTight)
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)
  const setEmoji = useEmojiFavicon((state) => state.setEmoji)

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isFocus && event.key === "w") {
        setZoomTight(!zoomTight)
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [zoomTight, isFocus, setZoomTight])

  const handleClick = () => {
    setZoomTight(!zoomTight)
    setEmoji(!zoomTight ? "🚦" : "⛰️")
  }

  return (
    <Tooltip title="Toggle zoom">
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px", position: "relative" }}
        onClick={handleClick}
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
