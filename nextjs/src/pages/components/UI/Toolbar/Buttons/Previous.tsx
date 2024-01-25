import Button from "@mui/material/Button"
import useCameraStore from "~/pages/hooks/useCameraStore"

export default function Previous() {
  const setCamera = useCameraStore((state) => state.setCamera)
  const previousCamera = useCameraStore((state) => state.previousCamera)

  const handleClick = () => {
    if (previousCamera) {
      setCamera(previousCamera)
    }
  }

  return (
    previousCamera !== null && (
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px" }}
        onClick={handleClick}
      >
        👈
      </Button>
    )
  )
}
