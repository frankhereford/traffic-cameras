import Image from "next/image"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"
import { useState, useEffect } from "react"
import styles from "./CctvCamera.module.css"

const markerSize = 5 // Half of the marker's size

const CctvCamera: React.FC = ({}) => {
  const [xRatio, setXRatio] = useState(1)
  const [yRatio, setYRatio] = useState(1)
  const camera = useIntersectionStore((state) => state.camera)
  const url = `https://cctv.austinmobility.io/image/${camera}.jpg`

  const [imageKey, setImageKey] = useState(Date.now())

  useEffect(() => {
    const timer = setTimeout(
      () => {
        setImageKey(Date.now()) // Change key to force re-render
      },
      5 * 60 * 1000,
    ) // 5 minutes

    return () => clearTimeout(timer) // Clear timeout if the component is unmounted
  }, [imageKey])

  const setCctvPendingPoint = useIntersectionStore(
    (state) => state.setCctvPendingPoint,
  )
  const cctvPendingPoint = useIntersectionStore(
    (state) => state.cctvPendingPoint,
  )
  const correlatedPoints = useIntersectionStore(
    (state) => state.correlatedPoints,
  )
  const setCctvImage = useIntersectionStore((state) => state.setCctvImage)

  const recognition = useIntersectionStore((state) => state.recognition)

  useEffect(() => {
    if (recognition !== null) {
      // Handle the recognition result here
      console.log("recognition: ", recognition)
    }
  }, [recognition])

  useEffect(() => {
    if (cctvPendingPoint === null) {
      setClickPosition(null)
    }
  }, [cctvPendingPoint])

  const [clickPosition, setClickPosition] = useState<{
    x: number
    y: number
  } | null>(null)

  const handleClick = (
    event: React.MouseEvent<HTMLImageElement, MouseEvent>,
  ) => {
    const img = event.currentTarget
    const rect = img.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    const xRatio = img.naturalWidth / img.width
    const yRatio = img.naturalHeight / img.height
    setXRatio(xRatio)
    setYRatio(yRatio)
    const nativeX = Math.floor(x * xRatio)
    const nativeY = Math.floor(y * yRatio)
    //console.log(`Clicked at native coordinates: ${nativeX}, ${nativeY}`)

    setCctvPendingPoint({
      x: nativeX,
      y: nativeY,
    })
    setClickPosition({
      x: nativeX / xRatio - markerSize,
      y: nativeY / yRatio - markerSize,
    })
  }

  const handleImageLoad = (event: React.SyntheticEvent<HTMLImageElement>) => {
    const img = event.currentTarget

    const canvas = document.createElement("canvas")
    const ctx = canvas.getContext("2d")

    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight

    ctx?.drawImage(img, 0, 0)

    const base64data = canvas.toDataURL("image/jpeg")
    //console.log("base64data", base64data)
    setCctvImage(base64data)
  }

  return (
    <>
      {camera ? (
        <div className={styles.relative}>
          <Image
            key={imageKey}
            priority
            src={url}
            width={1920}
            height={1080}
            alt="CCTV Camera"
            onClick={handleClick}
            onLoad={handleImageLoad}
          />
          {clickPosition && (
            <div
              className={styles.marker}
              style={{
                left: `${clickPosition.x}px`,
                top: `${clickPosition.y}px`,
              }}
            />
          )}

          {correlatedPoints.map((point, index) => (
            <div
              key={index}
              className={styles.marker}
              style={{
                left: `${point.cctvPoint.x / xRatio - markerSize}px`,
                top: `${point.cctvPoint.y / yRatio - markerSize}px`,
                backgroundColor: "blue",
              }}
            />
          ))}
        </div>
      ) : null}
    </>
  )
}

export default CctvCamera
