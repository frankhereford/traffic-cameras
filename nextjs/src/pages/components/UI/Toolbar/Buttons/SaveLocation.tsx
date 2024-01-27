import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import usePendingLocation from "~/pages/hooks/usePendingLocation"

export default function SaveLocation() {
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)
  const [showSaveButton, setShowSaveButton] = useState(false)

  const imageLocation = usePendingLocation((state) => state.imageLocation)
  const mapLocation = usePendingLocation((state) => state.mapLocation)

  useEffect(() => {
    if (imageLocation && mapLocation) {
      setShowSaveButton(true)
    } else {
      setShowSaveButton(false)
    }
  }, [imageLocation, mapLocation])

  const getCorrelatedLocation = usePendingLocation(
    (state) => state.getCorrelatedLocation,
  )

  const handleClick = () => {
    console.log("getCorrelatedLocation: ", getCorrelatedLocation())
    // if (previousCamera && !isFocus) {
    //   setCamera(previousCamera)
    // }
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

  const shouldRender = getCorrelatedLocation()

  if (!shouldRender) {
    return null
  }

  return (
    <Tooltip title="Go to previous camera">
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
