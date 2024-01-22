import { useSession } from "next-auth/react"
import DualPane from "~/pages/components/UI/DualPane"
import LandingPage from "../UI/LandingPage"

// import { api } from "~/utils/api";

export default function CameraGeoreferenceApp() {
  const { data: sessionData } = useSession()
  return <>{sessionData ? <DualPane /> : <LandingPage />}</>
}
