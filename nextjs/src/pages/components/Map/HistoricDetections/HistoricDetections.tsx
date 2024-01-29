import React, { useEffect } from "react"
import HistoricDetection from "~/pages/components/Map/HistoricDetections/HistoricDetection"

interface HistoricDetectionsProps {
  camera: number
}

import { api } from "~/utils/api"

export default function HistoricDetections({
  camera,
}: HistoricDetectionsProps) {
  const historicDetections = api.detection.getHistoricCameraDetections.useQuery(
    {
      camera: camera,
    },
  )

  useEffect(() => {
    if (historicDetections.data && historicDetections.data.length >= 2) {
      console.log(JSON.stringify(historicDetections.data.slice(0, 2), null, 2))
    }
  }, [historicDetections])

  return (
    <>
      {historicDetections.data?.map((image) =>
        image.detections.map((detection, index) => {
          if (
            typeof detection.latitude === "number" &&
            typeof detection.longitude === "number"
          ) {
            return (
              <HistoricDetection
                key={index}
                location={{
                  latitude: detection.latitude,
                  longitude: detection.longitude,
                }}
                label={detection.label}
                color={detection.isInsideConvexHull ? "green" : "red"}
              />
            )
          }
          return null
        }),
      )}
    </>
  )
}
