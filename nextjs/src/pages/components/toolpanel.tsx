import Auth from "./auth";
import Draggable, { DraggableCore } from "react-draggable";

export default function ToolPanel() {
  return (
    <>
      <Draggable>
        <div className="top-25 right-25 absolute z-50 w-36 rounded-lg bg-slate-200 pb-6 pt-6">
          <Auth />
        </div>
      </Draggable>
    </>
  );
}
