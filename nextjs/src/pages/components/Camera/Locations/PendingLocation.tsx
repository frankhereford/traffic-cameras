import React from "react"
import Typography from "@mui/material/Typography"

type Props = {
  location: { x: number; y: number } | null
}

export default function PendingLocation({ location }: Props) {
  if (!location) return <></>

  const coordinateStyle: React.CSSProperties = {
    pointerEvents: "none",
    position: "absolute",
    left: "5px",
    top: "5px",
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
      >{`${location.x}, ${location.y}`}</Typography>
    </div>
  )
}
