import * as React from "react"
import Box from "@mui/material/Box"
import Stack from "@mui/material/Stack"
import Slider from "@mui/material/Slider"
import useTransformedImage from "~/pages/hooks/useTransformedImage"

export default function TransparencySlider() {
  const [value, setValue] = React.useState<number>(30)
  const showTransformedImage = useTransformedImage(
    (state) => state.showTransformedImage,
  )
  const handleChange = (event: Event, newValue: number | number[]) => {
    setValue(newValue as number)
  }

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
