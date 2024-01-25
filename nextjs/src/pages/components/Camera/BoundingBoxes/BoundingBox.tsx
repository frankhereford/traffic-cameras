import React from "react"
import Typography from "@mui/material/Typography"
import { useTheme } from "@mui/material/styles"

interface BoundingBoxProps {
  id?: string
  label: string
  confidence: number
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  paneWidth: number
}

const BoundingBox: React.FC<BoundingBoxProps> = ({
  id,
  label,
  confidence,
  xMin,
  xMax,
  yMin,
  yMax,
  paneWidth,
}) => {
  const theme = useTheme()

  const originalImageWidth = 1920
  const scaleFactor =
    paneWidth < originalImageWidth ? paneWidth / originalImageWidth : 1

  const scaledXMin = xMin * scaleFactor
  const scaledXMax = xMax * scaleFactor
  const scaledYMin = yMin * scaleFactor
  const scaledYMax = yMax * scaleFactor

  const highlightedLabels = ["car", "truck"] // Add more labels as needed

  const boxStyle: React.CSSProperties = {
    position: "absolute",
    left: `${scaledXMin}px`,
    top: `${scaledYMin}px`,
    width: `${scaledXMax - scaledXMin}px`,
    height: `${scaledYMax - scaledYMin}px`,
    border: highlightedLabels.includes(label)
      ? `2px solid ${theme.palette.success.light}`
      : "1px solid grey",
    backgroundColor: "rgba(256, 256, 256, 0.05)",
  }

  const fontSize = 26

  return (
    <div style={boxStyle}>
      {/* <div>ID: {id}</div> */}
      <Typography
        style={{
          color: highlightedLabels.includes(label) ? "white" : "grey",
          fontSize: highlightedLabels.includes(label)
            ? `${fontSize * scaleFactor}px`
            : `${(fontSize * scaleFactor) / 2}px`,
          position: "relative",
          bottom: `${fontSize * scaleFactor + 5}px`,
          width: "300px",
          textShadow: highlightedLabels.includes(label)
            ? "-1px 0 black, 0 1px black, 1px 0 black, 0 -1px black"
            : "none",
          textTransform: "uppercase",
        }}
      >
        {label}
      </Typography>
      {/* <div>Confidence: {confidence}</div> */}
    </div>
  )
}

export default BoundingBox
