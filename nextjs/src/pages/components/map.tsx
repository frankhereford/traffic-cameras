import React, { useState, useCallback, useEffect } from "react"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"
import { GoogleMap, useJsApiLoader } from "@react-google-maps/api"

const Map: React.FC = ({}) => {
  const camera = useIntersectionStore((state) => state.camera)

  const [map, setMap] = useState<google.maps.Map | null>(null)

  const { isLoaded } = useJsApiLoader({
    id: "google-map-script",
    googleMapsApiKey: "AIzaSyAcbnyfHzwzLinnwjgapc7eMOg22yXkmuY",
  })

  const onUnmount = useCallback(() => {
    setMap(null)
  }, [])

  const containerStyle = {
    width: "100%",
    height: "100%",
  }

  const center = {
    lat: 30.2672,
    lng: -97.7431,
  }

  return isLoaded ? (
    <GoogleMap
      mapContainerStyle={containerStyle}
      center={center}
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
