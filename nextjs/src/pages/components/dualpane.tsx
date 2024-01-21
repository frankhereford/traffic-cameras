import { Allotment } from "allotment";
import "allotment/dist/style.css";
import ToolPanel from "./toolpanel";
import Map from "./map";
import Camera from "./camera";
import useApplicationStore from "../hooks/applicationstore";

export default function DualPane() {
  const setPaneWidths = useApplicationStore((state) => state.setPaneWidths);

  function onDrag(event: number[]) {
    // console.log("resizing", event);
    setPaneWidths(event);
  }

  return (
    <>
      <div style={{ height: "100vh", width: "100vw" }}>
        <ToolPanel />
        <Allotment onChange={onDrag} defaultSizes={[100, 100]}>
          <div
            className="bg-indigo-500"
            style={{
              background: "radial-gradient(at right top, #281450, #3A4957)",
            }}
          >
            <div style={{ height: "100vh" }}>
              <Camera />
            </div>
          </div>
          <div className="bg-slate-500">
            <div style={{ height: "100vh" }}>
              <Map />
            </div>
          </div>
        </Allotment>
      </div>
    </>
  );
}
