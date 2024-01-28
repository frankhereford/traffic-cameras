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

export default function ToolPanel() {
  return (
    <>
      <Draggable>
        <div className="top-25 right-25 mr-25 absolute z-50 w-[480px] rounded-lg bg-stone-50 pb-6 pl-[20px] pt-6">
          <div className="flex justify-center space-x-2">
            <ToggleMapFollow />
            <RandomCamera />
            <RandomNewCamera />
            <Previous />
            <SaveLocation />
            <ToggleLocations />
            <ResetLocations />
            <Logout />
          </div>
          <CameraPicker />
        </div>
      </Draggable>
    </>
  )
}
