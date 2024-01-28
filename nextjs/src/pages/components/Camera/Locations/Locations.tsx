import Location from "~/pages/components/Camera/Locations/Location"
import { api } from "~/utils/api"
import { getQueryKey } from "@trpc/react-query"
import { get } from "lodash"

interface LocationsProps {
  paneWidth: number
  camera: number
}

export default function Locations({ paneWidth, camera }: LocationsProps) {
  const locations = api.location.getLocations.useQuery({
    camera,
  })
  // const { data, isLoading, isError } = api.location.getLocations.useQuery({
  //   camera,
  // })

  // console.log("key?:", locations.getQueryCache)

  if (locations.isLoading) return <></>
  if (locations.isError) return <></>

  // console.log(getQueryKey(api.location.getLocations))

  // console.log("data:", data)
  // console.log("data:", JSON.stringify(locations.data, null, 2))

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
