import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useTransformedImage from "~/pages/hooks/useTransformedImage"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import useCameraStore from "~/pages/hooks/useCameraStore"
import useEmojiFavicon from "~/pages/hooks/useEmojiFavicon"

import { api } from "~/utils/api"
import useMapControls from "~/pages/hooks/useMapControls"

export default function ToggleTransformedImage() {
  const camera = useCameraStore((state) => state.camera)

  const showTransformedImage = useTransformedImage(
    (state) => state.showTransformedImage,
  )
  const setShowTransformedImage = useTransformedImage(
    (state) => state.setShowTransformedImage,
  )

  const zoomTight = useMapControls((state) => state.zoomTight)
  const setZoomTight = useMapControls((state) => state.setZoomTight)

  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)

  const setEmoji = useEmojiFavicon((state) => state.setEmoji)

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isFocus && event.key === "i") {
        setShowTransformedImage(!showTransformedImage)
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [showTransformedImage, isFocus, setShowTransformedImage])

  const locationCount = api.location.getLocationCount.useQuery(
    {
      camera: camera!,
    },
    {
      enabled: !!camera,
    },
  )

  const handleClick = () => {
    if (!zoomTight && !showTransformedImage) {
      setZoomTight(true)
    }
    setShowTransformedImage(!showTransformedImage)
    setEmoji(showTransformedImage ? "📐" : "📸")
  }

  if (locationCount.isLoading) return <></>
  if (locationCount.isError) return <></>
  if (locationCount.data <= 5) return <></>

  return (
    camera && (
      <Tooltip title="Toggle transformed image">
        <Button
          className="mb-4 p-0"
          variant="contained"
          style={{ fontSize: "35px", position: "relative" }}
          onClick={handleClick}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          {showTransformedImage ? "📸" : "📐"}
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
              i
            </span>
          )}
        </Button>
      </Tooltip>
    )
  )
}
