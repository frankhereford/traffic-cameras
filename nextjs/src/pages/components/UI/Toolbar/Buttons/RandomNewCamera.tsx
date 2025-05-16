/* eslint-disable @typescript-eslint/no-unused-vars */
import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import useCameraStore from "~/pages/hooks/useCameraStore"
import useGetSocrataData from "~/pages/hooks/useSocrataData"
import Badge from "@mui/material/Badge"
import Tooltip from "@mui/material/Tooltip"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import useEmojiFavicon from "~/pages/hooks/useEmojiFavicon"

import { api } from "~/utils/api"

export default function RandomNewCamera() {
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)

  const setCamera = useCameraStore((state) => state.setCamera)
  const setEmoji = useEmojiFavicon((state) => state.setEmoji)
  const {
    data: cameraData,
    isLoading: isCameraLoading,
    isError: isCameraError,
    error: cameraError,
  } = api.camera.getAllCameras.useQuery({})
  const {
    data: socrataData,
    isLoading: isSocrataLoading,
    isError: isSocrataError,
    error: socrataError,
  } = useGetSocrataData()

  const [newCameraCount, setNewCameraCount] = useState<number | undefined>(
    undefined,
  )

  useEffect(() => {
    if (socrataData && cameraData) {
      const cameraCoaIds = new Set(cameraData.map((camera) => camera.coaId))
      const newSocrataCameras = socrataData.filter(
        (camera) => !cameraCoaIds.has(parseInt(camera.camera_id)),
      )
      setNewCameraCount(newSocrataCameras.length)
    }
  }, [socrataData, cameraData])

  const handleClick = () => {
    if (socrataData && cameraData) {
      const cameraCoaIds = new Set(cameraData.map((camera) => camera.coaId))
      const newSocrataCameras = socrataData.filter(
        (camera) => !cameraCoaIds.has(parseInt(camera.camera_id)),
      )

      if (newSocrataCameras.length > 0) {
        const randomCamera =
          newSocrataCameras[
            Math.floor(Math.random() * newSocrataCameras.length)
          ]
        if (randomCamera && !isFocus) {
          setCamera(parseInt(randomCamera.camera_id))
        }
      }
    }
    setEmoji("🛰️")
  }

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "d") {
        handleClick()
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [socrataData, cameraData, isFocus])

  if (newCameraCount == 0) {
    return <></>
  }

  return (
    <>
      {cameraData && socrataData && (
        <Tooltip title="Random new camera">
          <Badge badgeContent={newCameraCount} color="primary" max={1000}>
            <Button
              className="mb-4 p-0"
              variant="contained"
              style={{ fontSize: "35px" }}
              onClick={handleClick}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
            >
              🛰️
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
                  d
                </span>
              )}
            </Button>
          </Badge>
        </Tooltip>
      )}
    </>
  )
}
