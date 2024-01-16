// import { set } from "zod"
import { useEffect, useState } from "react"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"
import { Button } from "~/pages/ui/button"

import { api } from "~/utils/api"

type Coordinate = {
  lat: number
  lng: number
}

const LockInPoints: React.FC = ({}) => {
  const mapPendingPoint = useIntersectionStore((state) => state.mapPendingPoint)
  const cctvPendingPoint = useIntersectionStore(
    (state) => state.cctvPendingPoint,
  )

  const setCctvPendingPoint = useIntersectionStore(
    (state) => state.setCctvPendingPoint,
  )
  const setMapPendingPoint = useIntersectionStore(
    (state) => state.setMapPendingPoint,
  )
  const correlatedPoints = useIntersectionStore(
    (state) => state.correlatedPoints,
  )
  const setCorrelatedPoints = useIntersectionStore(
    (state) => state.setCorrelatedPoints,
  )
  const setWarpedLabels = useIntersectionStore((state) => state.setWarpedLabels)

  const recognition = useIntersectionStore((state) => state.recognition)

  const warpCoordinates = api.transformation.warpCoordinates.useMutation({})
  const warpPendingCoordinates = api.transformation.warpCoordinates.useMutation(
    {},
  )
  const setMapPeekPoint = useIntersectionStore((state) => state.setMapPeekPoint)

  useEffect(() => {
    if (warpCoordinates.data) {
      setWarpedLabels(warpCoordinates.data)
    }
  }, [warpCoordinates.data])

  useEffect(() => {
    if (warpPendingCoordinates.data) {
      console.log("found peek point: ", warpPendingCoordinates.data)
      setMapPeekPoint(warpPendingCoordinates.data[0])
      //setWarpedLabels(warpCoordinates.data)
    }
  }, [warpPendingCoordinates.data])

  useEffect(() => {
    if (correlatedPoints.length > 4 && recognition) {
      // console.log("correlatedPoints", JSON.stringify(correlatedPoints, null, 2))

      console.log("recognition: ", JSON.stringify(recognition, null, 2))

      const distilledRecognition = recognition.Labels.filter(
        (label) => label.Name === "Car" && label.Instances.length > 0,
      ).flatMap((label) =>
        label.Instances.map((instance) => {
          const { BoundingBox } = instance
          const imageWidth = 1920
          const imageHeight = 1080

          // Calculate the middle of the x interval
          const x = (BoundingBox.Left + BoundingBox.Width / 2) * imageWidth
          // Calculate the minimum of the y interval
          const y = (BoundingBox.Top + BoundingBox.Height) * imageHeight
          return { x, y }
        }),
      )

      console.log("distilledRecognition: ")
      console.log(JSON.stringify(distilledRecognition, null, 2))

      if (distilledRecognition.length > 0) {
        warpCoordinates.mutate({
          points: correlatedPoints,
          labels: distilledRecognition,
        })
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [correlatedPoints, recognition])

  useEffect(() => {
    if (
      correlatedPoints.length > 4 &&
      cctvPendingPoint &&
      cctvPendingPoint.x !== 0 &&
      cctvPendingPoint.y !== 0
    ) {
      console.log(
        `Clicked at native coordinates: ${cctvPendingPoint.x}, ${cctvPendingPoint.y}`,
      )
      const cctvPendingPointArray: Coordinate[] = [
        {
          x: cctvPendingPoint.x,
          y: cctvPendingPoint.y,
        },
      ]
      warpPendingCoordinates.mutate({
        points: correlatedPoints,
        labels: cctvPendingPointArray as unknown as { x: number; y: number }[],
      })
    }
  }, [cctvPendingPoint, correlatedPoints])

  const resetPoints = () => {
    setCctvPendingPoint(null)
    setMapPendingPoint(null)
    setCorrelatedPoints([])
  }

  const addPoint = () => {
    if (cctvPendingPoint && mapPendingPoint) {
      setCorrelatedPoints([
        ...correlatedPoints,
        { cctvPoint: cctvPendingPoint, mapPoint: mapPendingPoint },
      ])
      setCctvPendingPoint(null)
      setMapPendingPoint(null)
      // console.log("correlatedPoints", JSON.stringify(correlatedPoints, null, 2))
    }
  }

  return (
    <>
      <div>CCTV Point: {cctvPendingPoint ? "✅" : "❌"}</div>
      <div>Map Point: {mapPendingPoint ? "✅" : "❌"}</div>
      <Button onClick={resetPoints}>Reset Points</Button>
      <Button
        onClick={addPoint}
        disabled={!cctvPendingPoint || !mapPendingPoint}
      >
        Add Point
      </Button>
    </>
  )
}

export default LockInPoints
