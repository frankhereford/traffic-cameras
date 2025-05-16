// eslint-disable-next-line @typescript-eslint/no-unused-vars
import CameraLocations from "./Locations/CameraLocations"
import useGetSocrataData from "~/pages/hooks/useSocrataData"
import useBoundingBox from "~/pages/hooks/useMapBoundingBox"
import { useMapControls } from "~/pages/hooks/useMapControls"
import { useCameraStore } from "~/pages/hooks/useCameraStore"
import React, { useState, useEffect, useCallback, useRef } from "react"
import type { SocrataData } from "~/pages/hooks/useSocrataData"
import usePendingLocation from "~/pages/hooks/usePendingLocation"
import { GoogleMap, useJsApiLoader, OverlayView } from "@react-google-maps/api"
import Locations from "~/pages/components/Map/Locations/Locations"
import useShowHistoricData from "~/pages/hooks/useShowHistoricData"
import Detections from "~/pages/components/Map/Detections/Detections"
import PendingLocation from "~/pages/components/Map/Locations/PendingLocation"
import HistoricDetections from "~/pages/components/Map/HistoricDetections/HistoricDetections"
import GeoreferencedImage from "~/pages/components/Map/GeoreferencedImage/GeoreferencedImage"
import { useSession } from "next-auth/react"

import { api } from "~/utils/api"

interface MapProps {
  paneWidth: number
  socrataData: SocrataData[]
}

const containerStyle = {
  width: "100%",
  height: "100%",
}

const maxZoom = 20

function Map({ socrataData, paneWidth }: MapProps) {
  const { isLoaded } = useJsApiLoader({
    id: "google-map-script",
    googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAP_API_KEY ?? "",
  })

  const [map, setMap] = useState<google.maps.Map | null>(null)
  const [center, setCenter] = useState<google.maps.LatLng | null>(null)
  const [bounds, setBounds] = useState<google.maps.LatLngBounds | null>(null)
  const camera = useCameraStore((state) => state.camera)
  const zoomTight = useMapControls((state) => state.zoomTight)
  const [pendingMapLocation, setPendingMapLocation] = useState<{
    latitude: number
    longitude: number
  } | null>(null)

  const setPendingMapLocationStore = usePendingLocation(
    (state) => state.setPendingMapLocation,
  )

  const showHistoricData = useShowHistoricData(
    (state) => state.showHistoricData,
  )

  const mapLocation = usePendingLocation((state) => state.mapLocation)

  const { data: session } = useSession()
  const isLoggedIn = !!session

  const [flashLocation, setFlashLocation] = useState<{ lat: number; lng: number } | null>(null)
  const flashTimer = useRef<NodeJS.Timeout | null>(null)
  const [flashZoom, setFlashZoom] = useState<number | null>(null) // Track zoom at flash start

  useEffect(() => {
    if (!zoomTight && map) {
      map.setZoom(zoomTight ? maxZoom : 14)
    }
  }, [zoomTight, map])

  const boundingBoxXMin = useBoundingBox((state) => state.xMin)
  const boundingBoxXMax = useBoundingBox((state) => state.xMax)
  const boundingBoxYMin = useBoundingBox((state) => state.yMin)
  const boundingBoxYMax = useBoundingBox((state) => state.yMax)

  useEffect(() => {
    if (
      map &&
      boundingBoxXMin &&
      boundingBoxXMax &&
      boundingBoxYMin &&
      boundingBoxYMax
    ) {
      const bounds = new google.maps.LatLngBounds(
        new google.maps.LatLng(boundingBoxYMin, boundingBoxXMin),
        new google.maps.LatLng(boundingBoxYMax, boundingBoxXMax),
      )
      if (zoomTight) {
        map.fitBounds(bounds)
      }
    }
  }, [
    map,
    boundingBoxXMin,
    boundingBoxXMax,
    boundingBoxYMin,
    boundingBoxYMax,
    zoomTight,
  ])

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

      if (cameraData?.location?.coordinates) {
        const latitude = cameraData.location.coordinates[1]
        const longitude = cameraData.location.coordinates[0]

        const location = new google.maps.LatLng(latitude!, longitude)
        if (zoomTight) {
          map.setZoom(maxZoom)
          map.panTo(location)
        } else {
          // Show a red flash at the camera location
          setFlashLocation({ lat: latitude!, lng: longitude! })
          setFlashZoom(map.getZoom() ?? null) // Store zoom at flash start
          if (flashTimer.current) clearTimeout(flashTimer.current)
        }
      }
      setBounds(map?.getBounds() ?? null)
      // Reset the pending map location when the camera changes
      setPendingMapLocation(null)
      setPendingMapLocationStore(null)
    }
  }, [camera, map, data, zoomTight, setPendingMapLocationStore])

  // Fade out the flash when zoom increases by 2 or more levels
  useEffect(() => {
    if (flashLocation && flashZoom !== null && map) {
      const currentZoom = map.getZoom() ?? 0
      if (currentZoom >= flashZoom + 2) {
        setFlashLocation(null)
        setFlashZoom(null)
      }
    }
  }, [map?.getZoom(), flashLocation, flashZoom, map])

  const locationCount = api.location.getLocationCount.useQuery(
    {
      camera: camera!,
    },
    {
      enabled: !!camera,
    },
  )

  useEffect(() => {
    if (locationCount.data) {
      console.log(`Location count: ${locationCount.data}`)
    }
  }, [locationCount.data])

  const CameraFlash = ({ location }: { location: { lat: number; lng: number } }) => {
    const [opacity, setOpacity] = useState(1)
    useEffect(() => {
      setOpacity(1)
    }, [location])
    return (
      <OverlayView
        position={location}
        mapPaneName={OverlayView.OVERLAY_MOUSE_TARGET}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            pointerEvents: "none",
            transform: "translate(16px, -50%)", // 16px right, vertically centered
            position: "absolute",
            top: "50%",
            left: 0,
            fontSize: 72, // Doubled size
            filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.2))",
            userSelect: "none",
          }}
        >
          <span role="img" aria-label="point right">👈</span>
        </div>
      </OverlayView>
    )
  }

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
      onClick={(e) => {
        if (!isLoggedIn) return
        if (e.latLng) {
          setPendingMapLocation({
            latitude: e.latLng.lat(),
            longitude: e.latLng.lng(),
          })
          setPendingMapLocationStore({
            latitude: e.latLng.lat(),
            longitude: e.latLng.lng(),
          })
        }
      }}
    >
      {bounds && socrataData && (
        <CameraLocations
          bounds={bounds}
          socrataData={socrataData}
          zoom={map?.getZoom()}
        />
      )}
      {flashLocation && !zoomTight && <CameraFlash location={flashLocation} />}
      {mapLocation && <PendingLocation location={pendingMapLocation} />}{" "}
      {camera && <Locations camera={camera} />}
      {camera && <Detections camera={camera} />}
      {camera && showHistoricData && <HistoricDetections camera={camera} />}
      {(locationCount.data ?? 0) > 5 && camera && (
        <GeoreferencedImage camera={camera} />
      )}
    </GoogleMap>
  ) : (
    <></>
  )
}

export default React.memo(Map)
