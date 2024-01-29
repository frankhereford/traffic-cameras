import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useShowHistoricData from "~/pages/hooks/useShowHistoricData"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"

export default function ToggleHistoricData() {
  const showHistoricData = useShowHistoricData(
    (state) => state.showHistoricData,
  )
  const setShowHistoricData = useShowHistoricData(
    (state) => state.setShowHistoricData,
  )
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isFocus && event.key === "h") {
        setShowHistoricData(!showHistoricData)
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [showHistoricData, isFocus, setShowHistoricData])

  return (
    <Tooltip title="Toggle historic data">
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px", position: "relative" }}
        onClick={() => setShowHistoricData(!showHistoricData)}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {showHistoricData ? "📚" : "✨"}
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
            h
          </span>
        )}
      </Button>
    </Tooltip>
  )
}
