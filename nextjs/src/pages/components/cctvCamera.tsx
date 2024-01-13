import Image from "next/image"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"

const CctvCamera: React.FC = ({}) => {
  const camera = useIntersectionStore((state) => state.camera)
  const url = `https://cctv.austinmobility.io/image/${camera}.jpg`

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
  }
  return (
    <>
      {camera ? (
        <Image
          priority
          src={url}
          width={1920}
          height={1080}
          alt="CCTV Camera"
          onClick={handleClick}
        />
      ) : null}
    </>
  )
}

export default CctvCamera
