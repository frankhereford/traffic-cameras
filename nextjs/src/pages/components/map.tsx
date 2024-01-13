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

  const containerStyle = {
    width: "100%",
    height: "100%",
  }

  return isLoaded ? (
    <GoogleMap
      mapContainerStyle={containerStyle}
      center={center ?? new google.maps.LatLng(30.2672, -97.7431)} // Use the center from state if it's defined, otherwise use a default value
      zoom={20}
      onUnmount={onUnmount}
      options={{ tilt: 0, mapTypeId: "satellite" }} // Set tilt to 0 for a top-down view
    />
  ) : (
    <></>
  )
}

export default Map
