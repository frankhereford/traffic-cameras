import React from "react"
import { Marker } from "@react-google-maps/api"

type Props = {
  location: { latitude: number; longitude: number } | null
}

export default function PendingLocation({ location }: Props) {
  if (!location) return null

  return (
    <Marker position={{ lat: location.latitude, lng: location.longitude }} />
  )
}
