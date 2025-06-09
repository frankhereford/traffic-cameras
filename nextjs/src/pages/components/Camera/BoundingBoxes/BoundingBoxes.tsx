import React, { useEffect, useState } from "react"
import { api } from "~/utils/api"
import BoundingBox from "~/pages/components/Camera/BoundingBoxes/BoundingBox"
import { useQueryClient } from "@tanstack/react-query"
import Stack from "@mui/material/Stack"
import CircularProgress from "@mui/material/CircularProgress"

interface BoundingBoxesProps {
  camera: number
  paneWidth: number
  imageWidth: number
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

const BoundingBoxes: React.FC<BoundingBoxesProps> = ({ camera, paneWidth, imageWidth }) => {
  const { data, isLoading, isError, error } = api.image.getDetections.useQuery({
    camera: camera,
  })
  const queryClient = useQueryClient()

  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null

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
  }, [data, queryClient])

  if (isLoading || !data) {
    return <></>
  }

  if (isError) {
    return <div>Error: {String(error)}</div>
  }

  const originalImageWidth = imageWidth
  const scaleFactor =
    paneWidth < originalImageWidth ? paneWidth / originalImageWidth : 1

  return (
    <>
      {data?.detectionsProcessed == false && (
        <div
          style={{
            position: "absolute",
            left: originalImageWidth * scaleFactor - 40, // 👈 the width of the spinner
            bottom: 0,
          }}
        >
          <Stack sx={{ color: "grey.500" }} spacing={2} direction="row">
            <CircularProgress color="success" />
          </Stack>
        </div>
      )}
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
          imageWidth={imageWidth}
        />
      ))}
    </>
  )
}

export default BoundingBoxes
