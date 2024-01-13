import React, { useState, useCallback, useEffect } from "react"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"
import { GoogleMap, useJsApiLoader } from "@react-google-maps/api"

const Map: React.FC = ({}) => {
  const cameraData = useIntersectionStore((state) => state.cameraData)

  const [map, setMap] = useState<google.maps.Map | null>(null)
  const [center, setCenter] = useState<{
    lat: number | undefined
    lng: number | undefined
  } | null>(null)

  const { isLoaded } = useJsApiLoader({
    id: "google-map-script",
    googleMapsApiKey: "AIzaSyAcbnyfHzwzLinnwjgapc7eMOg22yXkmuY",
  })

  useEffect(() => {
    if (cameraData?.location?.coordinates) {
      const newCenter = {
        lat: cameraData.location.coordinates[1],
        lng: cameraData.location.coordinates[0],
      }
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
      center={center ?? { lat: 30.2672, lng: -97.7431 }} // Use the center from state if it's defined, otherwise use a default value
      zoom={20}
      //onLoad={onLoad}
      onUnmount={onUnmount}
      options={{ tilt: 0, mapTypeId: "satellite" }} // Set tilt to 0 for a top-down view
    >
      {/* Child components, such as markers, info windows, etc. */}
      <></>
    </GoogleMap>
  ) : (
    <></>
  )
}

export default Map
