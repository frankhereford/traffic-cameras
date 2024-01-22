import { useEffect, useState } from "react"
import { useSession } from "next-auth/react"
import DualPane from "~/pages/components/UI/DualPane"
import LandingPage from "../UI/LandingPage"
import useGetSocrataData from "~/pages/hooks/useSocrataData"
import type { SocrataData } from "~/pages/hooks/useSocrataData"

export default function CameraGeoreferenceApp() {
  const { data: sessionData } = useSession()

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { data, isLoading, isError, error } = useGetSocrataData()

  useEffect(() => {
    console.log("Socrata Data:", { data, isLoading, isError, error })
  }, [data, isLoading, isError, error])

  const [storedData, setStoredData] = useState<SocrataData[] | null>(null)

  useEffect(() => {
    if (data) {
      setStoredData(data)
    }
  }, [data])

  return (
    <div
      style={{
        minHeight: "100vh",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {data && storedData && sessionData ? (
        <DualPane socrataData={storedData} />
      ) : (
        <LandingPage />
      )}
    </div>
  )
}
