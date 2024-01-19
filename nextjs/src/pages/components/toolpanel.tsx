import Auth from "./auth";
import CameraPicker from "./camerapicker";
import Draggable from "react-draggable";

export default function ToolPanel() {
  return (
    <>
      <Draggable>
        <div className="top-25 right-25 absolute z-50 w-[240px] rounded-lg bg-stone-50 pb-6 pl-[20px] pt-6">
          <CameraPicker />
          <Auth />
        </div>
      </Draggable>
    </>
  );
}
