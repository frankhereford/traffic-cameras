import React, { useState } from "react"
import { Marker, InfoWindow } from "@react-google-maps/api"

type Props = {
  location: { latitude: number; longitude: number } | null
}

export default function PendingLocation({ location }: Props) {
  const [isOpen, setIsOpen] = useState(false)

  if (!location) return <></>

  return (
    <Marker
      position={{ lat: location.latitude, lng: location.longitude }}
      icon={{
        url: "http://maps.google.com/mapfiles/ms/icons/grey.png",
      }}
      onClick={() => setIsOpen(true)}
    >
      {isOpen && (
        <InfoWindow onCloseClick={() => setIsOpen(false)}>
          <div>
            {location.latitude.toFixed(5)}, {location.longitude.toFixed(5)}
          </div>
        </InfoWindow>
      )}
    </Marker>
  )
}
