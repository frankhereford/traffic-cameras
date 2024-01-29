import Image from "next/image" // Import Image from Next.js
import { useEffect, useState, useCallback } from "react"
import useCameraStore from "~/pages/hooks/useCameraStore"
import { useQueryClient } from "@tanstack/react-query"
import BoundingBoxes from "~/pages/components/Camera/BoundingBoxes/BoundingBoxes"
import usePendingLocation from "~/pages/hooks/usePendingLocation"
import PendingLocation from "./Locations/PendingLocation"
import ReloadProgress from "./UI/ReloadProgress"
import Locations from "~/pages/components/Camera/Locations/Locations"
import { FullScreen, useFullScreenHandle } from "react-full-screen"

interface CameraProps {
  paneWidth: number
}

export default function Camera({ paneWidth }: CameraProps) {
  const camera = useCameraStore((state) => state.camera)
  const [imageKey, setImageKey] = useState(Date.now())
  const queryClient = useQueryClient()
  // ? do i really need to keep this in state, or can i just use the store?
  const [pendingImageLocation, setPendingImageLocation] = useState<{
    x: number
    y: number
  } | null>(null)
  const [reloadInterval, setReloadInterval] = useState(5 * 60 * 1000) // Default to 5 minutes
  const [reloadPercentage, setReloadPercentage] = useState(100)
  const fullScreenHandle = useFullScreenHandle()

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === "f") {
        fullScreenHandle.enter().catch((error) => {
          console.log("error: ", error)
        })
      }
    }

    window.addEventListener("keypress", handleKeyPress)

    // Cleanup - remove the event listener when the component is unmounted
    return () => {
      window.removeEventListener("keypress", handleKeyPress)
    }
  }, [fullScreenHandle])

  const setPendingImageLocationStore = usePendingLocation(
    (state) => state.setPendingImageLocation,
  )
  const setPendingMapLocation = usePendingLocation(
    (state) => state.setPendingMapLocation,
  )

  const imageLocation = usePendingLocation((state) => state.imageLocation)

  // make sure we always get a fresh image
  useEffect(() => {
    setImageKey(Date.now())
  }, [])

  // refresh the camera image on an interval
  useEffect(() => {
    const timer = setTimeout(() => {
      setImageKey(Date.now()) // Change key to force re-render
    }, reloadInterval)

    return () => clearTimeout(timer) // Clear timeout if the component is unmounted
  }, [imageKey, reloadInterval])

  // Update reloadPercentage every second
  useEffect(() => {
    const interval = setInterval(() => {
      const timeElapsed = Date.now() - imageKey
      const percentage = 100 - (timeElapsed / reloadInterval) * 100
      const clampedPercentage = Math.max(0, percentage)

      setReloadPercentage(clampedPercentage)
    }, 1000)

    return () => clearInterval(interval)
  }, [imageKey, reloadInterval])

  // invalidate certain queries when the image loads
  const handleImageLoad = () => {
    queryClient
      .invalidateQueries([["image", "getDetections"]])
      .catch((error) => {
        console.log("error: ", error)
      })
    queryClient
      .invalidateQueries([["detection", "getDetections"]])
      .catch((error) => {
        console.log("error: ", error)
      })
    // this is intended to do getCameras, getAllCameras, and getWorkingCameras
    queryClient.invalidateQueries([["camera"]]).catch((error) => {
      console.log("error: ", error)
    })

    // TODO: really, we should do this on camera change, not image load.
    setPendingImageLocation(null)
    setPendingImageLocationStore(null)
    setPendingMapLocation(null)
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

  const url = `http://flask:5000/image/${camera}?${imageKey}`
  return (
    <>
      <div>
        <FullScreen handle={fullScreenHandle}>
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
                <ReloadProgress progress={100 - reloadPercentage} />
                {imageLocation && pendingImageLocation && (
                  <PendingLocation
                    paneWidth={paneWidth}
                    location={pendingImageLocation}
                  />
                )}
                <BoundingBoxes camera={camera} paneWidth={paneWidth} />
                <Locations camera={camera} paneWidth={paneWidth} />
              </div>
            </>
          )}
        </FullScreen>
      </div>
    </>
  )
}
