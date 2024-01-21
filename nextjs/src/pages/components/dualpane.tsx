import { Allotment } from "allotment";
import "allotment/dist/style.css";
import ToolPanel from "./toolpanel";
import Map from "./map";
import Camera from "./camera";
import useApplicationStore from "../hooks/applicationstore";
import { useEffect, useRef, useState } from "react";
export default function DualPane() {
  const setPaneWidths = useApplicationStore((state) => state.setPaneWidths);
  const allotmentRef = useRef(null);
  const [toggle, setToggle] = useState(false);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "z") {
        setToggle(!toggle);
        const sizes = toggle ? [95, 5] : [5, 95];
        allotmentRef.current.resize(sizes);
      } else if (event.key === "Z") {
        allotmentRef.current.resize([50, 50]);
      }
    }

    document.addEventListener("keydown", handleKeyDown);

    // Cleanup function to remove the event listener when the component unmounts
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [toggle]);

  function onDrag(event: number[]) {
    // console.log("resizing", event);
    setPaneWidths(event);
  }

  return (
    <>
      <div style={{ height: "100vh", width: "100vw" }}>
        <ToolPanel />
        <Allotment
          ref={allotmentRef}
          onChange={onDrag}
          defaultSizes={[100, 100]}
        >
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
