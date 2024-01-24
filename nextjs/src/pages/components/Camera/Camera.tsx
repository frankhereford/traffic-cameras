import Image from "next/image" // Import Image from Next.js
import { useEffect, useState } from "react"
import { useCameraStore } from "~/pages/hooks/useCameraStore"
import { useQueryClient } from "@tanstack/react-query"

interface CameraProps {
  paneWidth: number
}

export default function Camera({ paneWidth }: CameraProps) {
  const camera = useCameraStore((state) => state.camera)
  const [imageKey, setImageKey] = useState(Date.now())
  const queryClient = useQueryClient()

  // refresh the camera image on an interval
  useEffect(() => {
    const timer = setTimeout(
      () => {
        setImageKey(Date.now()) // Change key to force re-render
      },
      5 * 60 * 1000, // 5 minutes
      // 30 * 1000, // 30 seconds
    )

    return () => clearTimeout(timer) // Clear timeout if the component is unmounted
  }, [imageKey])

  const url = `http://flask:5000/image/${camera}?${new Date().getTime()}`

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
          src={`${url}`}
          // src={`${url}?${new Date().getTime()}`}
          key={imageKey}
          priority
          alt="Camera Image"
          width={1920}
          height={1080}
          onLoad={handleImageLoad}
        />
      )}
    </>
  )
}
