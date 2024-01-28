import { useEffect, useState } from "react"
import { useLocationControls } from "~/pages/hooks/useLocationControls"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import { useCameraStore } from "~/pages/hooks/useCameraStore"

import { api } from "~/utils/api"

export default function ToggleLocations() {
  const showLocations = useLocationControls((state) => state.showLocations)
  const setShowLocations = useLocationControls(
    (state) => state.setShowLocations,
  )
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)
  const camera = useCameraStore((state) => state.camera)
  const locations = api.location.getLocations.useQuery(
    {
      camera: camera!,
    },
    {
      enabled: !!camera,
    },
  )
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isFocus && event.key === "p") {
        setShowLocations(!showLocations)
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [showLocations, isFocus, setShowLocations])

  if (locations.isLoading) return <></>
  if (locations.isError) return <></>
  if (locations.data.length === 0) return <></>

  return (
    <Tooltip title="Toggle correlated points">
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px", position: "relative" }}
        onClick={() => setShowLocations(!showLocations)}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {showLocations ? "📍" : "️🗺️"}
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
            p
          </span>
        )}
      </Button>
    </Tooltip>
  )
}
