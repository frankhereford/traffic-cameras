import * as React from "react"
import Box from "@mui/material/Box"
import Stack from "@mui/material/Stack"
import Slider from "@mui/material/Slider"
import useTransformedImage from "~/pages/hooks/useTransformedImage"
import useCameraStore from "~/pages/hooks/useCameraStore"

import { api } from "~/utils/api"

export default function TransparencySlider() {
  const camera = useCameraStore((state) => state.camera)

  const locationCount = api.location.getLocationCount.useQuery(
    {
      camera: camera!,
    },
    {
      enabled: !!camera,
    },
  )

  const [value, setValue] = React.useState<number>(30)
  const showTransformedImage = useTransformedImage(
    (state) => state.showTransformedImage,
  )

  const setOpacity = useTransformedImage((state) => state.setOpacity)

  const handleChange = (event: Event, newValue: number | number[]) => {
    setValue(newValue as number)
    setOpacity(newValue as number)
  }

  if (locationCount.isLoading) return <></>
  if (locationCount.isError) return <></>
  if (locationCount.data <= 5) return <></>
  if (!showTransformedImage) {
    return <></>
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Stack spacing={2} direction="row" sx={{ mb: 1 }} alignItems="center">
        <Slider aria-label="Volume" value={value} onChange={handleChange} />
      </Stack>
    </Box>
  )
}
