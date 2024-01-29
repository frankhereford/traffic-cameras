import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useShowHistoricData from "~/pages/hooks/useShowHistoricData"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"

import { api } from "~/utils/api"
import { useCameraStore } from "~/pages/hooks/useCameraStore"
import { sum } from "lodash"

export default function ToggleHistoricData() {
  const showHistoricData = useShowHistoricData(
    (state) => state.showHistoricData,
  )
  const setShowHistoricData = useShowHistoricData(
    (state) => state.setShowHistoricData,
  )
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)
  const camera = useCameraStore((state) => state.camera)

  const historicDetections = api.detection.getHistoricCameraDetections.useQuery(
    {
      camera: camera!,
    },
    {
      enabled: !!camera,
    },
  )

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isFocus && event.key === "h") {
        setShowHistoricData(!showHistoricData)
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [showHistoricData, isFocus, setShowHistoricData])

  const [sumOfValidDetections, setSumOfValidDetections] = useState(0)

  useEffect(() => {
    if (historicDetections.data) {
      const validLabels = [
        "car",
        "person",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
      ]
      const sum = historicDetections.data.reduce((sum, current) => {
        const validDetections = current.detections.filter(
          (detection) =>
            validLabels.includes(detection.label) &&
            detection.latitude !== null &&
            detection.longitude !== null,
        )
        return sum + validDetections.length
      }, 0)
      setSumOfValidDetections(sum)
    }
  }, [historicDetections.data])

  if (historicDetections.isLoading) return <></>
  if (historicDetections.isError) return <></>
  if (historicDetections.data?.length === 0) return <></>

  if (!sumOfValidDetections) return <></>

  return (
    <Tooltip title="Toggle historic data">
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px", position: "relative" }}
        onClick={() => setShowHistoricData(!showHistoricData)}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {showHistoricData ? "📚" : "✨"}
        {isHovered && (
          <span
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              fontSize: "50px",
              opacity: 0.15,
            }}
          >
            h
          </span>
        )}
      </Button>
    </Tooltip>
  )
}
