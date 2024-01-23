import Image from "next/image" // Import Image from Next.js
import { useCameraStore } from "~/pages/hooks/useCameraStore"
import { useQueryClient } from "@tanstack/react-query"

interface CameraProps {
  paneWidth: number
}

export default function Camera({ paneWidth }: CameraProps) {
  const camera = useCameraStore((state) => state.camera)
  const queryClient = useQueryClient()

  const url = `http://flask:5000/image/${camera}?timstamp=${Date.now()}`

  const handleImageLoad = () => {
    console.log("Image has finished loading")
    queryClient.invalidateQueries([["camera", "getCameras"]]).catch((error) => {
      console.log("error: ", error)
    })
  }

  return (
    <>
      {/* <div>{paneWidth}</div> */}
      {camera && (
        <Image
          src={url}
          alt="Camera Image"
          width={1920}
          height={1080}
          onLoad={handleImageLoad}
        />
      )}
    </>
  )
}
