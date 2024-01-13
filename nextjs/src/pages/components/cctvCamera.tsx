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

  useEffect(() => {
    console.log("correlatedPoints", JSON.stringify(correlatedPoints, null, 2))
  }, [correlatedPoints])

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
