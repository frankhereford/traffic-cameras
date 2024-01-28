import React, { useState } from "react"
import { Marker, InfoWindow } from "@react-google-maps/api"

type detectionProps = {
  location: { latitude: number; longitude: number }
  label: string
}

export default function Detection({ location, label }: detectionProps) {
  const [isOpen, setIsOpen] = useState(false)

  if (!location) return <></>

  return (
    <Marker
      position={{ lat: location.latitude, lng: location.longitude }}
      icon={{
        url: "http://maps.google.com/mapfiles/ms/icons/yellow.png",
      }}
      onClick={() => setIsOpen(true)}
    >
      {isOpen && (
        <InfoWindow onCloseClick={() => setIsOpen(false)}>
          <div>
            <div>{label}</div>
            <div>
              {location.latitude.toFixed(5)}, {location.longitude.toFixed(5)}
            </div>
          </div>
        </InfoWindow>
      )}
    </Marker>
  )
}
