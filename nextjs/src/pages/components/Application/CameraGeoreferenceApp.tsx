import { useEffect, useState } from "react"
import { useSession } from "next-auth/react"
import DualPane from "~/pages/components/UI/DualPane"
import LandingPage from "../UI/LandingPage"
import useGetSocrataData from "~/pages/hooks/useSocrataData"

export default function CameraGeoreferenceApp() {
  const { data, isLoading, isError, error } = useGetSocrataData()

  // Log the data states if needed
  useEffect(() => {
    console.log("Socrata Data:", { data, isLoading, isError, error })
  }, [data, isLoading, isError, error])

  const { data: sessionData } = useSession()

  return (
    <div
      style={{
        minHeight: "100vh",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {data && sessionData ? <DualPane /> : <LandingPage />}
    </div>
  )
}
