/* eslint-disable @typescript-eslint/no-unused-vars */
import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import useCameraStore from "~/pages/hooks/useCameraStore"
import useGetSocrataData from "~/pages/hooks/useSocrataData"
import Badge from "@mui/material/Badge"
import Tooltip from "@mui/material/Tooltip"

import { api } from "~/utils/api"

export default function RandomNewCamera() {
  const setCamera = useCameraStore((state) => state.setCamera)
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
        if (randomCamera) {
          setCamera(parseInt(randomCamera.camera_id))
        }
      }
    }
  }
  if (newCameraCount == 0) {
    return <></>
  }

  return (
    <>
      {cameraData && socrataData && (
        <Tooltip title="Save Location">
          <Badge badgeContent={newCameraCount} color="primary" max={1000}>
            <Button
              className="mb-4 p-0"
              variant="contained"
              style={{ fontSize: "35px" }}
              onClick={handleClick}
            >
              🛰️
            </Button>
          </Badge>
        </Tooltip>
      )}
    </>
  )
}
