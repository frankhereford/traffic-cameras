import React from "react"
import Typography from "@mui/material/Typography"

type Props = {
  location: { x: number; y: number } | null
  paneWidth: number
}

export default function PendingLocation({ location, paneWidth }: Props) {
  if (!location) return <></>

  // Understand that paneWidth never exceeds 1920
  paneWidth = paneWidth > 1920 ? 1920 : paneWidth

  const scaleFactor = paneWidth / 1920
  const scaledX = Math.round(location.x * scaleFactor)
  const scaledY = Math.round(location.y * scaleFactor)

  const coordinateStyle: React.CSSProperties = {
    position: "absolute",
    left: `${scaledX - 51}px`,
    top: `${scaledY - 4}px`,
    zIndex: 9999,
  }
  const textStyle: React.CSSProperties = {
    position: "absolute",
    bottom: "100%",
    left: "50%",
    transform: "translateX(-50%)",
    color: "white",
    fontSize: "14px",
    textShadow: "-1px 0 black, 0 1px black, 1px 0 black, 0 -1px black",
    textTransform: "uppercase",
    whiteSpace: "nowrap",
    overflow: "visible",
  }
  return (
    <div style={coordinateStyle}>
      <div style={textStyle}>
        <Typography>{`${location.x}, ${location.y}`}</Typography>
      </div>
      <svg height="100" width="100">
        <polygon points="50,0 60,40 50,30 40,40" fill="white" stroke="black" />
      </svg>
    </div>
  )
}
