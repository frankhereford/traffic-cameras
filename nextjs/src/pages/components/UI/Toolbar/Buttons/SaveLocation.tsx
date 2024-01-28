import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import usePendingLocation from "~/pages/hooks/usePendingLocation"
import useCameraStore from "~/pages/hooks/useCameraStore"
import { useQueryClient } from "@tanstack/react-query"

import { api } from "~/utils/api"

export default function SaveLocation() {
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)

  const imageLocation = usePendingLocation((state) => state.imageLocation)
  const mapLocation = usePendingLocation((state) => state.mapLocation)
  const [shouldRender, setShouldRender] = useState(false)

  const saveLocation = api.location.saveLocation.useMutation({})
  const camera = useCameraStore((state) => state.camera)
  const queryClient = useQueryClient()

  const setPendingImageLocation = usePendingLocation(
    (state) => state.setPendingImageLocation,
  )
  const setPendingMapLocation = usePendingLocation(
    (state) => state.setPendingMapLocation,
  )

  const getCorrelatedLocation = usePendingLocation(
    (state) => state.getCorrelatedLocation,
  )

  useEffect(() => {
    if (getCorrelatedLocation()) {
      setShouldRender(true)
    } else {
      setShouldRender(false)
    }
  }, [getCorrelatedLocation, imageLocation, mapLocation])

  const handleClick = () => {
    const correlatedLocation = getCorrelatedLocation()
    if (correlatedLocation && camera) {
      setPendingImageLocation(null)
      setPendingMapLocation(null)
      saveLocation.mutate(
        { correlatedLocation, camera },
        {
          onSuccess: () => {
            queryClient
              .invalidateQueries([["location", "getLocations"]])
              .catch((error) => {
                console.log("error: ", error)
              })
          },
        },
      )
    }
  }

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "s") {
        handleClick()
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isFocus])

  if (!shouldRender) {
    return <></>
  }

  return (
    <Tooltip title="Save Location">
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px", position: "relative" }}
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        📌
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
            s
          </span>
        )}
      </Button>
    </Tooltip>
  )
}
