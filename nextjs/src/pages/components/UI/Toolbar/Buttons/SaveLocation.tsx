import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import usePendingLocation from "~/pages/hooks/usePendingLocation"

export default function SaveLocation() {
  const [isHovered, setIsHovered] = useState(false)
  // if the autocomplete has focus, this turns true, so we can suppress the keyboard shortcut
  const isFocus = useAutocompleteFocus((state) => state.isFocus)

  const imageLocation = usePendingLocation((state) => state.imageLocation)
  const mapLocation = usePendingLocation((state) => state.mapLocation)
  const [shouldRender, setShouldRender] = useState(false)

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
    setPendingImageLocation(null)
    setPendingMapLocation(null)
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
