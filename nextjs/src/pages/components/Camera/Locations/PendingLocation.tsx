import React from "react"
import Typography from "@mui/material/Typography"

type Props = {
  location: { x: number; y: number } | null
  angle: number // angle in degrees
  paneWidth: number // width of the pane
}

export default function PendingLocation({ location, angle, paneWidth }: Props) {
  if (!location) return <></>

  const scaleFactor = paneWidth / 1920
  const scaledX = location.x * scaleFactor
  const scaledY = location.y * scaleFactor

  const coordinateStyle: React.CSSProperties = {
    pointerEvents: "none",
    position: "absolute",
    left: `${scaledX}px`,
    top: `${scaledY}px`,
  }

  return (
    <div style={coordinateStyle}>
      <Typography
        style={{
          color: "white",
          fontSize: "14px",
          position: "relative",
          textShadow: "-1px 0 black, 0 1px black, 1px 0 black, 0 -1px black",
          textTransform: "uppercase",
        }}
      >{`${scaledX.toFixed(2)}, ${scaledY.toFixed(2)}`}</Typography>
      <svg height="50" width="50">
        <polygon
          points="25,0 30,20 25,15 20,20"
          fill="red"
          transform={`rotate(${angle}, 25, 25)`}
        />
      </svg>
    </div>
  )
}
