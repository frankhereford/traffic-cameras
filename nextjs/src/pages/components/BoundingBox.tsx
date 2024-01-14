interface BoundingBoxProps {
  box: { left: number; top: number; width: number; height: number }
  image: {
    nativeWidth: number
    nativeHeight: number
    width: number
    height: number
  }
}

const BoundingBox: React.FC<BoundingBoxProps> = ({ box, image }) => {
  // Calculate the scale factors
  const xScale = image.width / image.nativeWidth
  const yScale = image.height / image.nativeHeight

  // Scale the bounding box
  const scaledBox = {
    left: box.left * xScale,
    top: box.top * yScale,
    width: box.width * xScale,
    height: box.height * yScale,
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
      }}
    />
  )
}

export default BoundingBox
