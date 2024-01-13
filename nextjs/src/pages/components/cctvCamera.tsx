import Image from "next/image"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"
import { useState } from "react"
import styles from "./CctvCamera.module.css"

const CctvCamera: React.FC = ({}) => {
  const camera = useIntersectionStore((state) => state.camera)
  const url = `https://cctv.austinmobility.io/image/${camera}.jpg`

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
    const nativeX = Math.floor(x * xRatio)
    const nativeY = Math.floor(y * yRatio)
    console.log(`Clicked at native coordinates: ${nativeX}, ${nativeY}`)
    const markerSize = 5 // Half of the marker's size
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
        </div>
      ) : null}
    </>
  )
}

export default CctvCamera
