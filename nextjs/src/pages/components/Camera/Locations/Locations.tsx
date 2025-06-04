import useLocationControls from "~/pages/hooks/useLocationControls"
import Location from "~/pages/components/Camera/Locations/Location"
import { api } from "~/utils/api"

interface LocationsProps {
  paneWidth: number
  camera: number
  imageWidth: number
}

export default function Locations({ paneWidth, camera, imageWidth }: LocationsProps) {
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
          location={{ x: location.x, y: location.y }}
          paneWidth={paneWidth}
          imageWidth={imageWidth}
        />
      ))}
    </>
  )
}
