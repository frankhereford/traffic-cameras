import CameraPicker from "~/pages/components/UI/Toolbar/CameraPicker"
import Draggable from "react-draggable"
import ToggleMapFollow from "~/pages/components/UI/Toolbar/Buttons/ToggleMapFollow"
import RandomCamera from "~/pages/components/UI/Toolbar/Buttons/RandomCamera"
import RandomNewCamera from "./Buttons/RandomNewCamera"
import Logout from "./Buttons/Logout"
import Login from "./Buttons/Login"
import Previous from "./Buttons/Previous"
import SaveLocation from "./Buttons/SaveLocation"
import ToggleLocations from "./Buttons/ToggleLocations"
import ResetLocations from "./Buttons/ResetLocations"
import ToggleHistoricData from "./Buttons/ToggleHistoricData"
import GitHub from "./Buttons/GitHub"
import ToggleTransformedImage from "./Buttons/ToggleTransformedImage"
import TransparencySlider from "./TransparencySlider"
import { useSession } from "next-auth/react"

export default function ToolPanel() {
  const { data: session } = useSession()
  return (
    <>
      <Draggable>
        <div
          className="top-25 right-25 mr-25 absolute z-50 rounded-lg bg-stone-50 pb-6 pr-[20px] pt-6"
          style={{ minWidth: "350px", maxWidth: "100%", paddingLeft: "20px" }}
        >
          <div className="flex flex-wrap justify-between space-x-2">
            <ToggleMapFollow />
            <RandomCamera />
            <RandomNewCamera />
            <Previous />
            <SaveLocation />
            <ToggleLocations />
            <ToggleHistoricData />
            <ToggleTransformedImage />
            <ResetLocations />
            {session ? <Logout /> : <Login />}
            <GitHub />
          </div>
          <TransparencySlider />
          <CameraPicker />
          <p
            className="pb-p mb-0 text-right font-sans text-xs text-gray-500"
            style={{ marginTop: "5px", marginBottom: "10px", height: "5px" }}
          >
            🙏🏻 &nbsp;
            <a
              href="https://data.austintexas.gov/"
              target="_blank"
              rel="noopener noreferrer"
              className="font-sans underline"
            >
              City of Austin, Open Data Portal
            </a>
          </p>
        </div>
      </Draggable>
    </>
  )
}
