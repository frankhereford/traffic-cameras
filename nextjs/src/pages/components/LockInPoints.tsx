// import { set } from "zod"
import { useEffect, useState } from "react"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"
import { Button } from "~/pages/ui/button"

import { api } from "~/utils/api"

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

  const recognition = useIntersectionStore((state) => state.recognition)

  const submitWarpRequest = api.transformation.submitWarpRequest.useMutation({})

  useEffect(() => {
    if (submitWarpRequest.data) {
      // Handle the result here
      console.log("submitWarpRequest", submitWarpRequest.data)
    }
  }, [submitWarpRequest.data])

  useEffect(() => {
    if (correlatedPoints.length > 4 && recognition) {
      console.log("asking for answers")
      console.log("correlatedPoints", JSON.stringify(correlatedPoints, null, 2))
      console.log("recognition", recognition)
      const distilledRecognition = recognition.Labels.filter(
        (label) => label.Instances.length > 0,
      ).map((label) => ({
        Name: label.Name,
        Confidence: label.Confidence,
        Instances: label.Instances, // Include the Instances
      }))
      console.log("distilledRecognition: ")
      console.log(JSON.stringify(distilledRecognition, null, 2))

      // submitWarpRequest.mutate({
      //   points: correlatedPoints,
      // })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [correlatedPoints, recognition])

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
