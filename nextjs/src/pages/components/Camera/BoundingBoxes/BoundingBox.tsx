import React from "react"
import Typography from "@mui/material/Typography"
import { useTheme } from "@mui/material/styles"

interface BoundingBoxProps {
  id: string
  label: string
  confidence: number
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  paneWidth: number
  imageWidth?: number
  imageHeight?: number
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
  imageWidth = 1920,
  imageHeight = 1080,
}) => {
  const theme = useTheme()

  // Use imageWidth for scaling instead of hardcoded value
  const scaleFactor =
    imageWidth && paneWidth < imageWidth ? paneWidth / imageWidth : 1

  const scaledXMin = xMin * scaleFactor
  const scaledXMax = xMax * scaleFactor
  const scaledYMin = yMin * scaleFactor
  const scaledYMax = yMax * scaleFactor

  const highlightedLabels = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    // "person",
  ]

  // Helper for case-insensitive comparison
  const isHighlighted = highlightedLabels
    .map(l => l.toLowerCase())
    .includes(label.toLowerCase())

  const boxStyle: React.CSSProperties = {
    pointerEvents: "none",
    position: "absolute",
    left: `${scaledXMin}px`,
    top: `${scaledYMin}px`,
    width: `${scaledXMax - scaledXMin}px`,
    height: `${scaledYMax - scaledYMin}px`,
    border: isHighlighted
      ? `2px solid ${theme.palette.success.light}`
      : "1px solid grey",
    backgroundColor: "rgba(256, 256, 256, 0.05)",
  }

  const fontSize = 23

  return (
    <div style={boxStyle}>
      {/* <div>ID: {id}</div> */}
      <Typography
        style={{
          color: isHighlighted ? "white" : "grey",
          fontSize: isHighlighted
            ? `${fontSize * scaleFactor}px`
            : `${(fontSize * scaleFactor) / 2}px`,
          position: "relative",
          bottom: `${fontSize * scaleFactor + 5}px`,
          width: "300px",
          textShadow: isHighlighted
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
