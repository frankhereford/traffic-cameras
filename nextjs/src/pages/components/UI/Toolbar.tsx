// import Auth from "./auth"
// import CameraPicker from "./camerapicker"
// import PickedPoints from "./pickedpoints"
import Draggable from "react-draggable"

export default function ToolPanel() {
  return (
    <>
      <Draggable>
        <div className="top-25 right-25 absolute z-50 w-[480px] rounded-lg bg-stone-50 pb-6 pl-[20px] pt-6">
          <div>hello</div>
          {/* <PickedPoints /> */}
          {/* <CameraPicker /> */}
          {/* <Auth /> */}
        </div>
      </Draggable>
    </>
  )
}
