import { useState, useEffect } from "react"
import type { SocrataData } from "~/pages/hooks/useSocrataData"
import { Marker } from "@react-google-maps/api"
import { useCameraStore } from "~/pages/hooks/useCameraStore"

interface CameraLocationsProps {
  socrataData: SocrataData[]
  bounds: google.maps.LatLngBounds
}

export default function CameraLocations({
  bounds,
  socrataData,
}: CameraLocationsProps) {
  const [filteredData, setFilteredData] = useState<SocrataData[]>([])
  const [markers, setMarkers] = useState<JSX.Element[] | null>()
  const setCamera = useCameraStore((state) => state.setCamera)

  useEffect(() => {
    if (socrataData && bounds) {
      const filtered = socrataData.filter((data) => {
        const { location } = data
        if (location?.coordinates) {
          const point = new google.maps.LatLng(
            location.coordinates[1]!,
            location.coordinates[0],
          )
          return bounds.contains(point)
        }
        return false
      })
      setFilteredData(filtered)
    }
  }, [socrataData, bounds])

  useEffect(() => {
    if (filteredData) {
      const markerElements = filteredData
        .map((data, index) => {
          const { location, camera_id } = data // Destructure camera_id from data
          if (location?.coordinates) {
            const position: google.maps.LatLng = new google.maps.LatLng(
              location.coordinates[1]!,
              location.coordinates[0],
            )
            return (
              <Marker
                key={index}
                position={position}
                icon={{
                  url: `http://maps.google.com/mapfiles/ms/icons/purple.png`,
                }}
                onClick={() => {
                  console.log(
                    `Marker with cameraId ${camera_id} was clicked.`,
                    setCamera(parseInt(camera_id)),
                  )
                }}
              />
            )
          }
          return null
        })
        .filter((element): element is JSX.Element => element !== null)
      setMarkers(markerElements)
    }
  }, [filteredData])

  return <>{markers}</>
}
