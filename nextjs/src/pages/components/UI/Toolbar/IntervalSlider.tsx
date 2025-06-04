import * as React from "react"
import { useEffect, useState } from "react"
import Box from "@mui/material/Box"
import Slider from "@mui/material/Slider"
import Typography from "@mui/material/Typography"
import useAutoRotateMode, { MIN_INTERVAL_S, MAX_INTERVAL_S } from "~/pages/hooks/useAutoRotateMode"

export default function IntervalSlider() {
  const autoRotateMode = useAutoRotateMode((state) => state.autoRotateMode)
  const autoRotateIntervalS = useAutoRotateMode((state) => state.autoRotateIntervalS)
  const setAutoRotateIntervalS = useAutoRotateMode((state) => state.setAutoRotateIntervalS)

  const [countdownSeconds, setCountdownSeconds] = useState(autoRotateIntervalS)

  useEffect(() => {
    // Reset countdown when the main interval changes
    setCountdownSeconds(autoRotateIntervalS)
  }, [autoRotateIntervalS])

  useEffect(() => {
    if (!autoRotateMode) {
      // Ensure countdown is reset if auto mode is turned off
      setCountdownSeconds(autoRotateIntervalS)
      return
    }

    if (countdownSeconds === 0) {
        // When countdown hits 0, the image should have rotated.
        // Reset countdown to the current interval for the next cycle.
        setCountdownSeconds(autoRotateIntervalS);
        // No need to return here, the timer below will continue if autoRotateMode is still true
    }

    const timer = setInterval(() => {
      setCountdownSeconds((prevSeconds) => {
        if (prevSeconds > 0) {
          return prevSeconds - 1
        }
        return 0; // Should be caught by the condition above and reset, but as a fallback.
      })
    }, 1000)

    return () => clearInterval(timer) // Cleanup timer
  }, [autoRotateMode, countdownSeconds, autoRotateIntervalS])

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
        Next image in: {countdownSeconds}s / {autoRotateIntervalS}s
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