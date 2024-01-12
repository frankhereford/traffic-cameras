/* eslint-disable @typescript-eslint/no-unsafe-return */
/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
import React, { useEffect } from "react"
import useCoordinateStore from "../hooks/CoordinateStore"

interface CCTVImageProps {
  cameraId: number
}

const CCTVImage: React.FC<CCTVImageProps> = ({ cameraId }) => {
  const cctv = "https://cctv.austinmobility.io/image/" + cameraId + ".jpg"

  const coordinates = useCoordinateStore((state) => state.coordinates)
  const addCoordinates = useCoordinateStore((state) => state.addCoordinates)

  useEffect(() => {
    console.log("coordinates: ", coordinates)
  }, [coordinates])

  const handleImageClick = (
    event: React.MouseEvent<HTMLImageElement, MouseEvent>,
  ) => {
    const imgElement = event.target as HTMLImageElement
    const originalWidth = imgElement.naturalWidth
    const originalHeight = imgElement.naturalHeight
    const currentWidth = imgElement.width
    const currentHeight = imgElement.height

    const xRatio = originalWidth / currentWidth
    const yRatio = originalHeight / currentHeight

    const originalX = event.clientX * xRatio
    const originalY = event.clientY * yRatio

    console.log(`Original X: ${originalX}, Original Y: ${originalY}`)
  }

  return (
    <img
      src={cctv}
      style={{ width: "50vw", height: "100vh", objectFit: "contain" }}
      onClick={handleImageClick}
    />
  )
}

export default CCTVImage
