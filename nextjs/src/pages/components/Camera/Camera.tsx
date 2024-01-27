import Image from "next/image" // Import Image from Next.js
import { useEffect, useState } from "react"
import useCameraStore from "~/pages/hooks/useCameraStore"
import { useQueryClient } from "@tanstack/react-query"
import BoundingBoxes from "~/pages/components/Camera/BoundingBoxes/BoundingBoxes"
import usePendingLocation from "~/pages/hooks/usePendingLocation"
import PendingLocation from "./Locations/PendingLocation"

interface CameraProps {
  paneWidth: number
}

export default function Camera({ paneWidth }: CameraProps) {
  const camera = useCameraStore((state) => state.camera)
  const [imageKey, setImageKey] = useState(Date.now())
  const queryClient = useQueryClient()
  const [pendingImageLocation, setPendingImageLocation] = useState<{
    x: number
    y: number
  } | null>(null)

  const setPendingImageLocationStore = usePendingLocation(
    (state) => state.setPendingImageLocation,
  )

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

  // invalidate certain queries when the image loads
  const handleImageLoad = () => {
    queryClient
      .invalidateQueries([["image", "getDetections"]])
      .catch((error) => {
        console.log("error: ", error)
      })
    queryClient.invalidateQueries([["camera", "getCameras"]]).catch((error) => {
      console.log("error: ", error)
    })
    queryClient
      .invalidateQueries([["camera", "getAllCameras"]])
      .catch((error) => {
        console.log("error: ", error)
      })
    queryClient
      .invalidateQueries([["camera", "getWorkingCameras"]])
      .catch((error) => {
        console.log("error: ", error)
      })
  }

  const handleImageClick = (
    event: React.MouseEvent<HTMLImageElement, MouseEvent>,
  ) => {
    const imgElement = event.currentTarget
    const scaleFactor = imgElement.naturalWidth / imgElement.clientWidth
    const x = Math.round(event.nativeEvent.offsetX * scaleFactor)
    const y = Math.round(event.nativeEvent.offsetY * scaleFactor)
    if (x === 0 || y === 0) {
      // ? this happens every 10th click or so .. why?
      return
    }

    setPendingImageLocation({ x, y })
    setPendingImageLocationStore({ x, y })
  }

  const url = `http://flask:5000/image/${camera}?${new Date().getTime()}`
  return (
    <>
      <div>
        {camera && (
          <>
            <div className="relative">
              <Image
                src={`${url}`}
                key={imageKey}
                priority
                alt="Camera Image"
                width={1920}
                height={1080}
                onLoad={handleImageLoad}
                onClick={handleImageClick}
              />
              <PendingLocation
                paneWidth={paneWidth}
                location={pendingImageLocation}
              />
              <BoundingBoxes camera={camera} paneWidth={paneWidth} />
            </div>
          </>
        )}
      </div>
    </>
  )
}
