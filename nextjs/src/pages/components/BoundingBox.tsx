import React, { useEffect } from "react"

interface BoundingBoxProps {
  box: { width: number; height: number; left: number; top: number }
  image: {
    nativeWidth: number
    nativeHeight: number
    width: number
    height: number
  }
  label?: string
}

const BoundingBox: React.FC<BoundingBoxProps> = ({ box, image, label }) => {
  // Calculate the scale factors
  const xScale = image.width / image.nativeWidth
  const yScale = image.height / image.nativeHeight

  // Scale the bounding box
  const scaledBox = {
    left: box.left * image.nativeWidth * xScale,
    top: box.top * image.nativeHeight * yScale,
    width: box.width * image.nativeWidth * xScale,
    height: box.height * image.nativeHeight * yScale,
  }

  return (
    <div
      style={{
        position: "absolute",
        left: `${scaledBox.left}px`,
        top: `${scaledBox.top}px`,
        width: `${scaledBox.width}px`,
        height: `${scaledBox.height}px`,
        border: "2px solid red",
        color: "white",
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        padding: "2px",
        fontSize: "10px", // Adjust this value to change the font size
      }}
    >
      {label}
    </div>
  )
}

export default BoundingBox
