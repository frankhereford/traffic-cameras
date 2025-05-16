import { useEffect, useState } from "react"
import { useSession } from "next-auth/react"
import DualPane from "~/pages/components/UI/DualPane"
import LandingPage from "../UI/LandingPage"
import useGetSocrataData from "~/pages/hooks/useSocrataData"
import useCameraStore from "~/pages/hooks/useCameraStore"
import type { SocrataData } from "~/pages/hooks/useSocrataData"

export default function CameraGeoreferenceApp() {
  const { data: sessionData } = useSession()

  // TODO i have started just using this hook when i need it and i should remove this prop drilling.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { data, isLoading, isError, error } = useGetSocrataData()

  const [storedData, setStoredData] = useState<SocrataData[] | null>(null)

  const setCamera = useCameraStore((state) => state.setCamera)
  const camera = useCameraStore((state) => state.camera)

  useEffect(() => {
    if (typeof window !== "undefined" && camera == null) {
      const stored = localStorage.getItem("lastViewedCamera")
      if (stored !== null) {
        const parsed = parseInt(stored)
        if (!isNaN(parsed)) setCamera(parsed)
      }
    }
  }, [camera, setCamera])

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
      {/* {data && storedData && sessionData ? ( */}
      {data && storedData ? (
        <DualPane socrataData={storedData} />
      ) : (
        <LandingPage />
      )}
    </div>
  )
}
