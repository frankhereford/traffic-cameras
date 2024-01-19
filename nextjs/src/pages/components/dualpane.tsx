import { Allotment } from "allotment";
import "allotment/dist/style.css";
import ToolPanel from "./toolpanel";
import Map from "./map";

export default function DualPane() {
  return (
    <>
      <div style={{ height: "100vh", width: "100vw" }}>
        <ToolPanel />
        <Allotment defaultSizes={[100, 100]}>
          <div
            className="bg-indigo-500"
            style={{
              background: "radial-gradient(at right top, #281450, #3A4957)",
            }}
          >
            <div style={{ height: "100vh" }}></div>
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
