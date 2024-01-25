import CameraPicker from "~/pages/components/UI/Toolbar/CameraPicker"
// import PickedPoints from "./pickedpoints"
import Draggable from "react-draggable"
import TightZoom from "~/pages/components/UI/Toolbar/Buttons/TightZoom"
import RandomCamera from "~/pages/components/UI/Toolbar/Buttons/RandomCamera"
import RandomNewCamera from "./Buttons/RandomNewCamera"
import Logout from "./Buttons/Logout"
import Previous from "./Buttons/Previous"

export default function ToolPanel() {
  return (
    <>
      <Draggable>
        <div className="top-25 right-25 absolute z-50 w-[480px] rounded-lg bg-stone-50 pb-6 pl-[20px] pt-6">
          {/* <PickedPoints /> */}
          <div className="flex justify-center space-x-4">
            <TightZoom />
            <RandomCamera />
            <RandomNewCamera />
            <Previous />
            <Logout />
          </div>
          <CameraPicker />
        </div>
      </Draggable>
    </>
  )
}
