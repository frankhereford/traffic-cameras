import React, { useState, useCallback, useEffect } from "react"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"
import { GoogleMap, useJsApiLoader } from "@react-google-maps/api"

const Map: React.FC = ({}) => {
  const cameraData = useIntersectionStore((state) => state.cameraData)

  const [map, setMap] = useState<google.maps.Map | null>(null)
  const [center, setCenter] = useState<google.maps.LatLng | null>(null)

  const { isLoaded } = useJsApiLoader({
    id: "google-map-script",
    googleMapsApiKey: "AIzaSyAcbnyfHzwzLinnwjgapc7eMOg22yXkmuY",
  })

  useEffect(() => {
    if (cameraData?.location?.coordinates) {
      const newCenter = new google.maps.LatLng(
        cameraData.location.coordinates[1]!,
        cameraData.location.coordinates[0],
      )
      setCenter(newCenter)
    }
  }, [map, cameraData])

  const onUnmount = useCallback(() => {
    setMap(null)
  }, [])

  const handleClick = (e: {
    latLng: { lat: () => unknown; lng: () => unknown }
  }) => {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
    const lat = e.latLng?.lat()
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
    const lng = e.latLng?.lng()
    // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
    console.log(`Clicked at ${lat}, ${lng}`)
  }

  const containerStyle = {
    width: "100%",
    height: "100%",
  }

  return isLoaded ? (
    <GoogleMap
      mapContainerStyle={containerStyle}
      center={center ?? new google.maps.LatLng(30.2672, -97.7431)}
      zoom={20}
      onUnmount={onUnmount}
      options={{ tilt: 0, mapTypeId: "satellite" }}
      onClick={handleClick}
    />
  ) : (
    <></>
  )
}

export default Map
