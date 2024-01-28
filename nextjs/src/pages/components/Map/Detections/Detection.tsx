/* eslint-disable @next/next/no-img-element */
import React, { useState } from "react"
import { Marker, InfoWindow } from "@react-google-maps/api"

type detectionProps = {
  location: { latitude: number; longitude: number }
  label: string
  picture: string
}

export default function Detection({
  location,
  label,
  picture,
}: detectionProps) {
  const [isOpen, setIsOpen] = useState(false)

  if (!location) return <></>

  return (
    <Marker
      position={{ lat: location.latitude, lng: location.longitude }}
      icon={{
        url: "http://maps.google.com/mapfiles/ms/icons/yellow.png",
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
            <div>
              <img
                src={`data:image/jpeg;base64,${picture}`}
                alt={label}
                style={{ maxWidth: "250px", maxHeight: "250px" }}
              />{" "}
            </div>
          </div>
        </InfoWindow>
      )}
    </Marker>
  )
}
