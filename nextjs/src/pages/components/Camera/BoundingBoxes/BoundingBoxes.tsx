import React, { useEffect } from "react"
import { api } from "~/utils/api"
import BoundingBox from "~/pages/components/Camera/BoundingBoxes/BoundingBox"
import { useQueryClient } from "@tanstack/react-query"

interface BoundingBoxesProps {
  camera: number
  paneWidth: number
}

interface Detection {
  id: string
  label: string
  confidence: number
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

const BoundingBoxes: React.FC<BoundingBoxesProps> = ({ camera, paneWidth }) => {
  const { data, isLoading, isError, error } = api.image.getDetections.useQuery({
    camera: camera,
  })
  const queryClient = useQueryClient()

  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null

    //if (data?.detections.length == 0) {
    if (data?.detectionsProcessed == false) {
      intervalId = setInterval(() => {
        // it's ugly to poll but it works
        queryClient
          .invalidateQueries([["image", "getDetections"]])
          .catch((error) => {
            console.log("error: ", error)
          })
      }, 1000) // Run every 1000 milliseconds (1 second)
    }

    // Clear interval on component unmount
    return () => {
      if (intervalId) {
        clearInterval(intervalId)
      }
    }
  }, [data])
  if (isLoading || !data) {
    return <></>
  }

  if (isError) {
    return <div>Error: {String(error)}</div>
  }

  return (
    <>
      {data.detections.map((detection: Detection) => (
        <BoundingBox
          key={detection.id}
          id={detection.id}
          label={detection.label}
          confidence={detection.confidence}
          xMin={detection.xMin}
          xMax={detection.xMax}
          yMin={detection.yMin}
          yMax={detection.yMax}
          paneWidth={paneWidth}
        />
      ))}
    </>
  )
}

export default BoundingBoxes
