import useIntersectionStore from "~/pages/hooks/IntersectionStore"
import { Button } from "~/pages/ui/button"

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

  const resetPoints = () => {
    setCctvPendingPoint(null)
    setMapPendingPoint(null)
  }

  const addPoint = () => {
    if (cctvPendingPoint && mapPendingPoint) {
      setCorrelatedPoints([
        ...correlatedPoints,
        { cctvPoint: cctvPendingPoint, mapPoint: mapPendingPoint },
      ])
      resetPoints()
      console.log("correlatedPoints", JSON.stringify(correlatedPoints, null, 2))
    }
  }

  return (
    <>
      <div>CCTV Point: {cctvPendingPoint ? "✅" : "❌"}</div>
      <div>Map Point: {mapPendingPoint ? "✅" : "❌"}</div>
      <Button onClick={resetPoints}>Reset Points</Button>
      <Button onClick={addPoint}>Add Point</Button>
    </>
  )
}

export default LockInPoints
