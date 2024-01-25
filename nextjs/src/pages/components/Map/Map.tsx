// eslint-disable-next-line @typescript-eslint/no-unused-vars
import React, { useState, useEffect, useCallback } from "react"
import { GoogleMap, useJsApiLoader } from "@react-google-maps/api"

import type { SocrataData } from "~/pages/hooks/useSocrataData"
import CameraLocations from "./CameraLocations"
import { useCameraStore } from "~/pages/hooks/useCameraStore"
import useGetSocrataData from "~/pages/hooks/useSocrataData"
import { useMapControls } from "~/pages/hooks/useMapControls"

interface MapProps {
  paneWidth: number
  socrataData: SocrataData[]
}

const containerStyle = {
  width: "100%",
  height: "100%",
}

const maxZoom = 20

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function Map({ socrataData, paneWidth }: MapProps) {
  const { isLoaded, loadError } = useJsApiLoader({
    id: "google-map-script",
    googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAP_API_KEY ?? "",
  })

  const [map, setMap] = useState<google.maps.Map | null>(null)
  const [center, setCenter] = useState<google.maps.LatLng | null>(null)
  const [bounds, setBounds] = useState<google.maps.LatLngBounds | null>(null)
  const camera = useCameraStore((state) => state.camera)
  const zoomTight = useMapControls((state) => state.zoomTight)

  useEffect(() => {
    if (!zoomTight && map) {
      map.setZoom(zoomTight ? maxZoom : 14)
    }
  }, [zoomTight, map])

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { data, isLoading, isError, error } = useGetSocrataData()

  const onUnmount = useCallback(function callback() {
    setMap(null)
  }, [])

  const onLoad = useCallback(function callback(map: google.maps.Map) {
    setMap(map)
  }, [])

  const OnViewportChange = () => {
    setBounds(map?.getBounds() ?? null)
  }

  useEffect(() => {
    if (isLoaded) {
      const [latitude, longitude] = [30.262531, -97.753983] // this will get replaced with the location out of useStorage or whatever
      setCenter(new google.maps.LatLng(latitude, longitude))
    }
  }, [isLoaded])

  useEffect(() => {
    if (camera && map && data) {
      const cameraData = data.find(
        (item) => parseInt(item.camera_id, 10) === camera,
      )

      if (cameraData ?? cameraData!.location.coordinates) {
        const latitude = cameraData!.location.coordinates[1]
        const longitude = cameraData!.location.coordinates[0]

        const location = new google.maps.LatLng(latitude!, longitude)
        if (zoomTight) {
          map.setZoom(maxZoom)
          map.panTo(location)
        } else {
          // map.setZoom(15)
        }
      }
      setBounds(map?.getBounds() ?? null)
    }
  }, [camera, map, data, zoomTight])

  return isLoaded ? (
    <GoogleMap
      mapContainerStyle={containerStyle}
      center={center ?? new google.maps.LatLng(30.262531, -97.753983)}
      zoom={17}
      onUnmount={onUnmount}
      onLoad={onLoad}
      onDragEnd={OnViewportChange}
      onZoomChanged={OnViewportChange}
      options={{ tilt: 0, mapTypeId: "satellite" }}
    >
      {bounds &&
        socrataData && ( // don't fire it off until it has the data it needs
          <CameraLocations bounds={bounds} socrataData={socrataData} />
        )}
    </GoogleMap>
  ) : (
    <></>
  )
}

export default React.memo(Map)
