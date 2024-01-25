import { useEffect, useState } from "react"
import useGetSocrataData from "~/pages/hooks/useSocrataData"
import type { SocrataData } from "~/pages/hooks/useSocrataData"
import Autocomplete from "@mui/material/Autocomplete"
import TextField from "@mui/material/TextField"
import { useCameraStore } from "~/pages/hooks/useCameraStore"

interface Camera {
  label: string
  cameraId: string
}

export default function CameraPicker() {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { data, isLoading, isError, error } = useGetSocrataData()
  const [cameras, setCameras] = useState<Camera[] | undefined>()
  const setCamera = useCameraStore((state) => state.setCamera)
  const camera = useCameraStore((state) => state.camera)

  useEffect(() => {
    if (data) {
      // todo order these by radiating distance from map center
      // todo filter these by the ones we know won't work
      const transformedData = data.map((item: SocrataData) => ({
        label: item.location_name.trim(),
        cameraId: item.camera_id,
      }))
      setCameras(transformedData)
    }
  }, [data])

  useEffect(() => {
    console.log("Cameras:", cameras)
  }, [cameras])

  if (isLoading) return <></>
  if (isError) return <>Error</>

  return (
    <>
      <Autocomplete
        disablePortal
        id="combo-box-demo"
        options={cameras!}
        sx={{ width: 440 }}
        renderInput={(params) => <TextField {...params} label="Camera" />}
        onChange={(event, value) => {
          if (value) {
            setCamera(parseInt(value.cameraId))
          }
        }}
      />
      {/* <Autocomplete disablePortal id="cameraPicker" options={cameras} /> */}
    </>
  )
}
