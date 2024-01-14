import React, { useState, useCallback, useEffect } from "react"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"
import { GoogleMap, useJsApiLoader, Marker } from "@react-google-maps/api"

const Map: React.FC = ({}) => {
  const cameraData = useIntersectionStore((state) => state.cameraData)

  const [map, setMap] = useState<google.maps.Map | null>(null)
  const [center, setCenter] = useState<google.maps.LatLng | null>(null)
  const [markerPosition, setMarkerPosition] =
    useState<google.maps.LatLng | null>(null)

  const { isLoaded } = useJsApiLoader({
    id: "google-map-script",
    googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY ?? "",
  })

  const setMapPendingPoint = useIntersectionStore(
    (state) => state.setMapPendingPoint,
  )

  const correlatedPoints = useIntersectionStore(
    (state) => state.correlatedPoints,
  )

  const mapPendingPoint = useIntersectionStore((state) => state.mapPendingPoint)

  useEffect(() => {
    if (mapPendingPoint === null) {
      // console.log("mapPendingPoint is null")
      setMarkerPosition(null)
    }
  }, [mapPendingPoint])

  useEffect(() => {
    if (isLoaded && cameraData?.location?.coordinates) {
      const newCenter = new google.maps.LatLng(
        cameraData.location.coordinates[1]!,
        cameraData.location.coordinates[0],
      )
      setCenter(newCenter)
    }
  }, [map, cameraData, isLoaded])

  const onUnmount = useCallback(() => {
    setMap(null)
  }, [])

  const handleClick = (e: {
    latLng: { lat: () => unknown; lng: () => unknown }
  }) => {
    const lat = e.latLng?.lat() as number
    const lng = e.latLng?.lng() as number
    // console.log(`Clicked at ${lat}, ${lng}`)
    setMapPendingPoint({ lat, lng })
    setMarkerPosition(new google.maps.LatLng(lat, lng))
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
    >
      {markerPosition && <Marker position={markerPosition} />}
      {correlatedPoints.map((point, index) => (
        <Marker
          key={index}
          position={
            new google.maps.LatLng(point.mapPoint.lat, point.mapPoint.lng)
          }
          icon={{
            url: "http://maps.google.com/mapfiles/ms/icons/blue-dot.png",
          }}
        />
      ))}
    </GoogleMap>
  ) : (
    <></>
  )
}

export default Map
