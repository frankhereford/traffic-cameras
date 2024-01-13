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

  const resetPoints = () => {
    setCctvPendingPoint(null)
    setMapPendingPoint(null)
  }

  return (
    <>
      <div>CCTV Point: {cctvPendingPoint ? "✅" : "❌"}</div>
      <div>Map Point: {mapPendingPoint ? "✅" : "❌"}</div>
      <Button onClick={resetPoints}>Reset Points</Button>
    </>
  )
}

export default LockInPoints
