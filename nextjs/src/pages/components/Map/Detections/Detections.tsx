import React, { useEffect } from "react"
import { api } from "~/utils/api"
import { useQueryClient } from "@tanstack/react-query"
import Detection from "./Detection"
import useBoundingBox from "~/pages/hooks/useMapBoundingBox"

interface DetectionProps {
  camera: number
}

export default function Detections({ camera }: DetectionProps) {
  const queryClient = useQueryClient()

  const detectedObjects = api.detection.getDetections.useQuery({
    camera: camera,
  })

  const setBoundingBox = useBoundingBox((state) => state.setBoundingBox)

  useEffect(() => {
    const validLabels = [
      "car",
      "person",
      "bus",
      "truck",
      "bicycle",
      "motorcycle",
    ]

    if (detectedObjects.data?.detections) {
      const validDetections = detectedObjects.data.detections.filter((d) =>
        validLabels.includes(d.label),
      )

      const latitudes = validDetections.map((d) => d.latitude).filter(Boolean)
      const longitudes = validDetections.map((d) => d.longitude).filter(Boolean)

      if (latitudes.length > 0 && longitudes.length > 0) {
        const xMin = Math.min(...longitudes)
        const xMax = Math.max(...longitudes)
        const yMin = Math.min(...latitudes)
        const yMax = Math.max(...latitudes)

        setBoundingBox(xMin, xMax, yMin, yMax)
      }
    }
  }, [detectedObjects.data?.detections, setBoundingBox])
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null

    if (detectedObjects.data?.detectionsProcessed == false) {
      intervalId = setInterval(() => {
        // it's still ugly to poll but it works
        queryClient
          .invalidateQueries([["detection", "getDetections"]])
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
  }, [detectedObjects, queryClient])

  if (
    !detectedObjects.data ||
    !detectedObjects.data?.detections ||
    detectedObjects.data?.detectionsProcessed == false
  ) {
    return <></>
  }

  const validLabels = ["car", "person", "bus", "truck", "bicycle", "motorcycle"]

  return (
    <>
      {detectedObjects.data.detections.map((detection) =>
        typeof detection.latitude === "number" &&
        typeof detection.longitude === "number" &&
        typeof detection.picture === "string" &&
        validLabels.includes(detection.label) ? (
          <Detection
            key={detection.id}
            label={detection.label}
            picture={detection.picture}
            color={detection.isInsideConvexHull ? "yellow" : "grey"}
            location={{
              latitude: detection.latitude,
              longitude: detection.longitude,
            }}
          />
        ) : null,
      )}
    </>
  )
}
