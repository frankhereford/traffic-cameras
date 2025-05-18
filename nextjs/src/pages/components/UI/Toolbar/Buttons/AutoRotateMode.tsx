import { useEffect, useRef, useState, useCallback } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useCameraStore from "~/pages/hooks/useCameraStore"
import useAutoRotateMode from "~/pages/hooks/useAutoRotateMode"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import useEmojiFavicon from "~/pages/hooks/useEmojiFavicon"
import { api } from "~/utils/api"

const AUTO_ROTATE_INTERVAL_MS = 15000

export default function AutoRotateMode() {
  
  const autoRotateMode = useAutoRotateMode(
    (state) => state.autoRotateMode,
  )
  const setAutoRotateMode = useAutoRotateMode(
    (state) => state.setAutoRotateMode,
  )

  // const [autoMode, setAutoMode] = useState(false)
  const [isHovered, setIsHovered] = useState(false)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const setCamera = useCameraStore((state) => state.setCamera)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)
  const setEmoji = useEmojiFavicon((state) => state.setEmoji)

  // Use the same query as RandomCamera
  const { data } = api.camera.getWorkingCameras.useQuery({})

  // Pick a random camera from the working set
  const pickRandomCamera = useCallback(() => {
    if (data && data.length > 0 && !isFocus) {
      const randomCamera = data[Math.floor(Math.random() * data.length)]
      if (randomCamera) {
        setCamera(randomCamera.coaId)
        setEmoji("🔀")
      }
    }
  }, [data, isFocus, setCamera, setEmoji])

  useEffect(() => {
    console.log("AutoRotateMode effect: ", autoRotateMode)
    if (autoRotateMode) {
      intervalRef.current = setInterval(() => {
        pickRandomCamera()
      }, AUTO_ROTATE_INTERVAL_MS)
    }
  }, [autoRotateMode])

  const handleClick = () => {
    setAutoRotateMode(!autoRotateMode)
  }

  if (!data || data.length === 0) return <></>

  return (
    <Tooltip title="Toggle auto-random camera (every 15s)">
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px", position: "relative" }}
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        color={autoRotateMode ? "success" : "primary"}
      >
        {autoRotateMode ? "🔁" : "▶️"}
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
            a
          </span>
        )}
      </Button>
    </Tooltip>
  )
}
