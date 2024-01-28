import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useCameraStore from "~/pages/hooks/useCameraStore"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import { useQueryClient } from "@tanstack/react-query"
import Modal from "@mui/material/Modal"
import Box from "@mui/material/Box"
import Typography from "@mui/material/Typography"
import { useTheme } from "@mui/material/styles"

import { api } from "~/utils/api"

export default function Previous() {
  const theme = useTheme()
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

  const [open, setOpen] = useState(false)

  const handleOpen = () => setOpen(true)
  const handleClose = () => setOpen(false)

  const handleConfirmClick = () => {
    handleClose()
    handleClick()
  }

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
        handleOpen()
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
    <>
      <Tooltip title="Reset correlated points">
        <Button
          className="mb-4 p-0"
          variant="contained"
          style={{ fontSize: "35px", position: "relative" }}
          onClick={handleOpen}
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
      <Modal open={open} onClose={handleClose}>
        <Box
          sx={{
            width: 400,
            padding: 2,
            bgcolor: "background.paper",
            margin: "auto",
            marginTop: "15%",
            borderRadius: 2,
          }}
        >
          <Typography variant="h6" component="h2">
            Confirm Reset
          </Typography>
          <Typography sx={{ mt: 2 }}>
            Are you sure you want to reset correlated points?
          </Typography>
          <Button
            variant="contained"
            color="primary"
            onClick={handleConfirmClick}
            sx={{ mt: 2 }}
            style={{
              background: theme.palette.primary.main,
            }}
          >
            Confirm
          </Button>
          <Button
            variant="outlined"
            onClick={handleClose}
            sx={{ mt: 2, ml: 2 }}
          >
            Cancel
          </Button>
        </Box>
      </Modal>
    </>
  )
}
