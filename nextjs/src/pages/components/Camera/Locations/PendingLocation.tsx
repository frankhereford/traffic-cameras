import React from "react"
import Typography from "@mui/material/Typography"

type Props = {
  location: { x: number; y: number } | null
  paneWidth: number // width of the pane
}

export default function PendingLocation({ location, paneWidth }: Props) {
  if (!location) return <></>

  // Ensure paneWidth never exceeds 1920
  paneWidth = paneWidth > 1920 ? 1920 : paneWidth

  const scaleFactor = paneWidth / 1920
  const scaledX = location.x * scaleFactor
  const scaledY = location.y * scaleFactor

  const coordinateStyle: React.CSSProperties = {
    pointerEvents: "none",
    position: "absolute",
    left: `${scaledX - 25}px`,
    top: `${scaledY - 5}px`,
  }

  return (
    <>
      <Typography
        style={{
          color: "white",
          fontSize: "14px",
          position: "relative",
          textShadow: "-1px 0 black, 0 1px black, 1px 0 black, 0 -1px black",
          textTransform: "uppercase",
        }}
      >{`${scaledX.toFixed(2)}, ${scaledY.toFixed(2)}`}</Typography>
      <div style={coordinateStyle}>
        <svg height="50" width="50">
          <polygon points="25,0 30,20 25,15 20,20" fill="red" />
        </svg>
      </div>
    </>
  )
}
