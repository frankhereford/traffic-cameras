import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import useCameraStore from "~/pages/hooks/useCameraStore"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import Tooltip from "@mui/material/Tooltip"

import { api } from "~/utils/api"

export default function RandomCamera() {
  const setCamera = useCameraStore((state) => state.setCamera)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { data, isLoading, isError, error } =
    api.camera.getWorkingCameras.useQuery({})

  const [isHovered, setIsHovered] = useState(false)

  const handleClick = () => {
    if (data && !isFocus) {
      const randomCamera = data[Math.floor(Math.random() * data.length)]!
      const cameraId = randomCamera.coaId
      setCamera(cameraId)
    }
  }

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "r") {
        handleClick()
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, isFocus])

  return (
    <>
      {data && data.length > 0 && (
        <Tooltip title="Random camera">
          <Button
            className="mb-4 p-0"
            variant="contained"
            style={{ fontSize: "35px", position: "relative" }}
            onClick={handleClick}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
          >
            🎯
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
                r
              </span>
            )}
          </Button>
        </Tooltip>
      )}
    </>
  )
}
