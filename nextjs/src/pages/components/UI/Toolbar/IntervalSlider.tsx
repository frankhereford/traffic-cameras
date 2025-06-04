import * as React from "react"
import Box from "@mui/material/Box"
import Slider from "@mui/material/Slider"
import Typography from "@mui/material/Typography"
import useAutoRotateMode, { MIN_INTERVAL_S, MAX_INTERVAL_S } from "~/pages/hooks/useAutoRotateMode"

export default function IntervalSlider() {
  const autoRotateMode = useAutoRotateMode((state) => state.autoRotateMode)
  const autoRotateIntervalS = useAutoRotateMode((state) => state.autoRotateIntervalS)
  const setAutoRotateIntervalS = useAutoRotateMode((state) => state.setAutoRotateIntervalS)

  if (!autoRotateMode) {
    return <></>
  }

  const handleChange = (event: Event, newValue: number | number[]) => {
    // Prevent click from propagating to map or other elements
    if (event.stopPropagation) {
      event.stopPropagation();
    }
    setAutoRotateIntervalS(newValue as number)
  }

  return (
    <Box sx={{ width: '100%', px: 1, mt: 1, mb: 1 }}>
      <Typography id="interval-slider-label" gutterBottom sx={{ fontSize: '0.75rem', textAlign: 'center', color: 'text.secondary' }}>
        Auto-Rotate Interval: {autoRotateIntervalS}s
      </Typography>
      <Slider
        value={autoRotateIntervalS}
        min={MIN_INTERVAL_S}
        max={MAX_INTERVAL_S}
        step={1}
        onChange={handleChange}
        aria-labelledby="interval-slider-label"
        size="small"
        onMouseDown={(e) => e.stopPropagation()} // Prevent map drag
      />
    </Box>
  )
} 