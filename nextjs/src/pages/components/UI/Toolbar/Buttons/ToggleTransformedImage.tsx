import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useTransformedImage from "~/pages/hooks/useTransformedImage"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import useCameraStore from "~/pages/hooks/useCameraStore"

import { api } from "~/utils/api"

export default function ToggleTransformedImage() {
  const camera = useCameraStore((state) => state.camera)

  const showTransformedImage = useTransformedImage(
    (state) => state.showTransformedImage,
  )
  const setShowTransformedImage = useTransformedImage(
    (state) => state.setShowTransformedImage,
  )

  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)

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
          onClick={() => setShowTransformedImage(!showTransformedImage)}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          {showTransformedImage ? "📸" : "💨"}
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
