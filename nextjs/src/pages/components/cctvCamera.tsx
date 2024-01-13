import Image from "next/image"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"

const CctvCamera: React.FC = ({}) => {
  const camera = useIntersectionStore((state) => state.camera)
  const url = `https://cctv.austinmobility.io/image/${camera}.jpg`

  return (
    <>
      {camera ? (
        <Image src={url} width={1920} height={1080} alt="CCTV Camera" />
      ) : null}
    </>
  )
}

export default CctvCamera
