import Button from "@mui/material/Button"
import useCameraStore from "~/pages/hooks/useCameraStore"

import { api } from "~/utils/api"

export default function RandomCamera() {
  const setCamera = useCameraStore((state) => state.setCamera)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { data, isLoading, isError, error } =
    api.camera.getWorkingCameras.useQuery({})

  const dataString = data ? JSON.stringify(data, null, 2) : ""
  console.log(dataString)

  const handleClick = () => {
    if (data) {
      const randomCamera = data[Math.floor(Math.random() * data.length)]!
      const cameraId = randomCamera.coaId
      setCamera(cameraId)
    }
  }

  return (
    <>
      {data && data.length > 0 && (
        <Button
          className="mb-4 p-0"
          variant="contained"
          style={{ fontSize: "35px" }}
          onClick={handleClick}
        >
          🎯
        </Button>
      )}
    </>
  )
}
