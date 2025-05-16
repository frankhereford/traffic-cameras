import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useCameraStore from "~/pages/hooks/useCameraStore"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import useEmojiFavicon from "~/pages/hooks/useEmojiFavicon"

export default function Previous() {
  const setCamera = useCameraStore((state) => state.setCamera)
  const previousCamera = useCameraStore((state) => state.previousCamera)
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)
  const setEmoji = useEmojiFavicon((state) => state.setEmoji)

  const handleClick = () => {
    if (previousCamera && !isFocus) {
      setCamera(previousCamera)
      setEmoji("👈")
    }
  }

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "b") {
        handleClick()
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [previousCamera, isFocus])

  return (
    previousCamera !== null && (
      <Tooltip title="Go to previous camera">
        <Button
          className="mb-4 p-0"
          variant="contained"
          style={{ fontSize: "35px", position: "relative" }}
          onClick={handleClick}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          👈
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
              b
            </span>
          )}
        </Button>
      </Tooltip>
    )
  )
}
