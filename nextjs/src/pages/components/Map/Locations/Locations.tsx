import useLocationControls from "~/pages/hooks/useLocationControls"
import Location from "~/pages/components/Map/Locations/Location"
import { api } from "~/utils/api"

interface LocationsProps {
  camera: number
}

export default function Locations({ camera }: LocationsProps) {
  const locations = api.location.getLocations.useQuery({
    camera,
  })
  const showLocations = useLocationControls((state) => state.showLocations)

  if (locations.isLoading) return <></>
  if (locations.isError) return <></>
  if (!showLocations) return <></>

  return (
    <>
      {locations.data.map((location) => (
        <Location
          key={location.id}
          location={{
            latitude: location.latitude,
            longitude: location.longitude,
          }}
        />
      ))}
    </>
  )
}
