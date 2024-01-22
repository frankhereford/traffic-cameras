import { useSession } from "next-auth/react"
import DualPane from "~/pages/components/UI/DualPane"
import LandingPage from "../UI/LandingPage"

import { useQuery } from "@tanstack/react-query"
import { useEffect } from "react"

async function fetchSocrataData() {
  const url = "https://data.austintexas.gov/resource/b4k4-adkb.json"

  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const data = await response.json()
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return data
  } catch (error) {
    // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
    throw new Error(`Failed to fetch data: ${error}`)
  }
}

export default function CameraGeoreferenceApp() {
  const socrataData = useQuery({
    queryKey: ["socrataData"],
    queryFn: fetchSocrataData,
  })

  useEffect(() => {
    console.log("")
    console.log("socrataData.isFetching", socrataData.isFetching)
    console.log("socrataData.isLoading", socrataData.isLoading)
    console.log("socrataData.isError", socrataData.isError)
    console.log("socrataData.data", socrataData.data)
    console.log("socrataData.error", socrataData.error)
    console.log("")
  }, [socrataData])

  const { data: sessionData } = useSession()
  return <>{sessionData ? <DualPane /> : <LandingPage />}</>
}
