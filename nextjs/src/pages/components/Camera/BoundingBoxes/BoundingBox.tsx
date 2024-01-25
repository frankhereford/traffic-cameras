import React from "react"

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
  const originalImageWidth = 1920
  const scaleFactor =
    paneWidth < originalImageWidth ? paneWidth / originalImageWidth : 1

  const scaledXMin = xMin * scaleFactor
  const scaledXMax = xMax * scaleFactor
  const scaledYMin = yMin * scaleFactor
  const scaledYMax = yMax * scaleFactor

  const boxStyle: React.CSSProperties = {
    position: "absolute",
    left: `${scaledXMin}px`,
    top: `${scaledYMin}px`,
    width: `${scaledXMax - scaledXMin}px`,
    height: `${scaledYMax - scaledYMin}px`,
    border: "2px solid blue",
    backgroundColor: "rgba(256, 256, 256, 0.05)",
  }

  return (
    <div style={boxStyle}>
      {/* <div>ID: {id}</div> */}
      {label}
      {/* <div>Confidence: {confidence}</div> */}
    </div>
  )
}

export default BoundingBox
