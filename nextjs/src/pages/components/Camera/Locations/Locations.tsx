import Location from "~/pages/components/Camera/Locations/Location"
import { api } from "~/utils/api"

interface LocationsProps {
  paneWidth: number
  camera: number
}

export default function Locations({ paneWidth, camera }: LocationsProps) {
  const locations = api.location.getLocations.useQuery({
    camera,
  })

  if (locations.isLoading) return <></>
  if (locations.isError) return <></>

  return (
    <>
      {locations.data.map((location) => (
        <Location
          key={location.id}
          location={{ x: location.x, y: location.y }}
          paneWidth={paneWidth}
        />
      ))}
    </>
  )
}
