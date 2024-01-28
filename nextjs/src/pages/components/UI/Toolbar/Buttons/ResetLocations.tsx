import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useCameraStore from "~/pages/hooks/useCameraStore"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import { useQueryClient } from "@tanstack/react-query"

import { api } from "~/utils/api"

export default function Previous() {
  const camera = useCameraStore((state) => state.camera)
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)
  const resetLocations = api.location.resetLocations.useMutation({})
  const locations = api.location.getLocations.useQuery(
    {
      camera: camera!,
    },
    {
      enabled: !!camera,
    },
  )
  const queryClient = useQueryClient()

  const handleClick = () => {
    if (camera) {
      resetLocations.mutate(
        { camera: camera },
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
      if (event.key === "e") {
        handleClick()
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, isFocus])

  if (locations.isLoading) return <></>
  if (locations.isError) return <></>
  if (locations.data.length === 0) return <></>

  return (
    <Tooltip title="Reset correlated points">
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px", position: "relative" }}
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        🧹
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
            e
          </span>
        )}
      </Button>
    </Tooltip>
  )
}
