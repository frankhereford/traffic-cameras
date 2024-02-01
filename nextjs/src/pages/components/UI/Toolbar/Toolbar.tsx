import CameraPicker from "~/pages/components/UI/Toolbar/CameraPicker"
import Draggable from "react-draggable"
import ToggleMapFollow from "~/pages/components/UI/Toolbar/Buttons/ToggleMapFollow"
import RandomCamera from "~/pages/components/UI/Toolbar/Buttons/RandomCamera"
import RandomNewCamera from "./Buttons/RandomNewCamera"
import Logout from "./Buttons/Logout"
import Previous from "./Buttons/Previous"
import SaveLocation from "./Buttons/SaveLocation"
import ToggleLocations from "./Buttons/ToggleLocations"
import ResetLocations from "./Buttons/ResetLocations"
import ToggleHistoricData from "./Buttons/ToggleHistoricData"
import GitHub from "./Buttons/GitHub"
import ToggleTransformedImage from "./Buttons/ToggleTransformedImage"

export default function ToolPanel() {
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
            <Logout />
            <GitHub />
          </div>
          <CameraPicker />
        </div>
      </Draggable>
    </>
  )
}
