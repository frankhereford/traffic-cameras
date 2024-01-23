import { useState, useEffect } from "react"
import type { SocrataData } from "~/pages/hooks/useSocrataData"
import { Marker } from "@react-google-maps/api"

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
          const { location } = data
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
