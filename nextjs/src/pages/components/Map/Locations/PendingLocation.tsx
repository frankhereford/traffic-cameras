import React from "react"
import { Marker } from "@react-google-maps/api"

type Props = {
  location: { latitude: number; longitude: number } | null
}

export default function PendingLocation({ location }: Props) {
  if (!location) return <></>

  return (
    <Marker
      position={{ lat: location.latitude, lng: location.longitude }}
      icon={{
        url: "http://maps.google.com/mapfiles/ms/icons/grey.png",
      }}
    />
  )
}
