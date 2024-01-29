import React, { useState } from "react"
import { Marker, InfoWindow } from "@react-google-maps/api"

type historicDetectionProps = {
  location: { latitude: number; longitude: number }
  label: string
  color?: string
}

export default function HistoricDetection({
  location,
  label,
  color = "yellow",
}: historicDetectionProps) {
  const [isOpen, setIsOpen] = useState(false)

  if (!location) return <></>

  return (
    <Marker
      position={{ lat: location.latitude, lng: location.longitude }}
      icon={{
        url: `http://maps.google.com/mapfiles/ms/icons/${color}.png`,
      }}
      onClick={() => setIsOpen(!isOpen)}
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
