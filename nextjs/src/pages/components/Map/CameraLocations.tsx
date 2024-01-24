import { useState, useEffect } from "react"
import type { SocrataData } from "~/pages/hooks/useSocrataData"
import { Marker } from "@react-google-maps/api"
import { useCameraStore } from "~/pages/hooks/useCameraStore"
// import { getQueryKey } from "@trpc/react-query"

import { api } from "~/utils/api"

interface CameraLocationsProps {
  socrataData: SocrataData[]
  bounds: google.maps.LatLngBounds
}

// Outside of your component
const statusColors: Record<string, string> = {
  ok: "green.png",
  "404": "red.png",
  unavailable: "yellow.png",
}
export default function CameraLocations({
  bounds,
  socrataData,
}: CameraLocationsProps) {
  const [filteredData, setFilteredData] = useState<SocrataData[]>([])
  const [markers, setMarkers] = useState<JSX.Element[] | null>()
  const setCamera = useCameraStore((state) => state.setCamera)
  const [cameraMap, setCameraMap] = useState<Record<number, string>>({})

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { data, isLoading, isError, error } = api.camera.getCameras.useQuery({
    cameras: filteredData.map((data) => parseInt(data.camera_id)),
  })

  // const cameraKey = getQueryKey(api.camera.getCameras, undefined, "any")
  // console.log("cameraKey: ", JSON.stringify(cameraKey, null, 2))

  useEffect(() => {
    if (data) {
      const queriedCameraMap: Record<number, string> = {}
      data.forEach((camera) => {
        console.log(camera)
        queriedCameraMap[camera.coaId] = camera.status!.name
      })
      console.log("queriedCameraMap: ", queriedCameraMap)
      setCameraMap(queriedCameraMap)
    }
  }, [data])

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
                  url: `http://maps.google.com/mapfiles/ms/icons/${statusColors[cameraMap[parseInt(camera_id)]!] ?? "purple.png"}`,
                }}
                onClick={() => {
                  // console.log(`Marker with cameraId ${camera_id} was clicked.`)
                  setCamera(parseInt(camera_id))
                }}
              />
            )
          }
          return null
        })
        .filter((element): element is JSX.Element => element !== null)
      setMarkers(markerElements)
    }
  }, [filteredData, cameraMap, setCamera])

  return <>{markers}</>
}
