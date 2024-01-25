// import Auth from "./auth"
import CameraPicker from "~/pages/components/UI/Toolbar/CameraPicker"
// import PickedPoints from "./pickedpoints"
import Draggable from "react-draggable"
import TightZoom from "~/pages/components/UI/Toolbar/BooleanSettings/TightZoom"

export default function ToolPanel() {
  return (
    <>
      <Draggable>
        <div className="top-25 right-25 absolute z-50 w-[480px] rounded-lg bg-stone-50 pb-6 pl-[20px] pt-6">
          {/* <PickedPoints /> */}
          <TightZoom />
          <CameraPicker />
          {/* <Auth /> */}
        </div>
      </Draggable>
    </>
  )
}
