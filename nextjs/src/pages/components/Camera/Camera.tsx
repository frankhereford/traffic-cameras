import Image from "next/image" // Import Image from Next.js
import { useCameraStore } from "~/pages/hooks/useCameraStore"

interface CameraProps {
  paneWidth: number
}

export default function Camera({ paneWidth }: CameraProps) {
  const camera = useCameraStore((state) => state.camera)

  const url = `http://flask:5000/image/${camera}?timstamp=${Date.now()}`

  return (
    <>
      {/* <div>{paneWidth}</div> */}
      {camera && (
        <Image
          src={url}
          alt="Camera Image"
          width={1920} // Replace with your desired width
          height={1080} // Replace with your desired height
        />
      )}
    </>
  )
}
