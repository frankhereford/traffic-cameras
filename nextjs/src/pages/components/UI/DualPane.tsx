import { useEffect, useRef, useState } from "react"
import { Allotment } from "allotment"
import type { AllotmentHandle } from "allotment"
import "allotment/dist/style.css"
// import ToolPanel from "./toolpanel";
import Camera from "~/pages/components/Camera/Camera"
import Map from "~/pages/components/Map/Map"

import type { SocrataData } from "~/pages/hooks/useSocrataData"
import Toolbar from "./Toolbar/Toolbar"

import { debounce } from "lodash"

interface DualPaneProps {
  socrataData: SocrataData[]
}

export default function DualPane({ socrataData }: DualPaneProps) {
  const [toggle, setToggle] = useState(false)
  const allotmentRef = useRef<AllotmentHandle | null>(null)
  const cameraDivRef = useRef<HTMLDivElement | null>(null)
  const mapDivRef = useRef<HTMLDivElement | null>(null)
  const [cameraPaneWidth, setCameraPaneWidth] = useState(0)
  const [mapPaneWidth, setMapPaneWidth] = useState(0)

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "z") {
        setToggle(!toggle)
        const sizes = toggle ? [95, 5] : [5, 95]
        allotmentRef.current?.resize(sizes)
      } else if (event.key === "Z") {
        allotmentRef.current?.resize([50, 50])
      }
    }

    document.addEventListener("keydown", handleKeyDown)

    // Cleanup function to remove the event listener when the component unmounts
    return () => {
      document.removeEventListener("keydown", handleKeyDown)
    }
  }, [toggle])

  useEffect(() => {
    setCameraPaneWidth(cameraDivRef.current?.offsetWidth ?? 0)
    setMapPaneWidth(mapDivRef.current?.offsetWidth ?? 0)
  }, [])

  const debouncedOnDrag = debounce(() => {
    setCameraPaneWidth(cameraDivRef.current?.offsetWidth ?? 0)
    setMapPaneWidth(mapDivRef.current?.offsetWidth ?? 0)
  }, 250) // 250ms delay

  return (
    <>
      <div style={{ height: "100vh", width: "100vw" }}>
        <Toolbar />
        <Allotment
          ref={allotmentRef}
          onChange={debouncedOnDrag}
          defaultSizes={[100, 100]}
        >
          <div
            ref={cameraDivRef}
            className="bg-indigo-500"
            style={{
              background: "radial-gradient(at right top, #281450, #3A4957)",
              // borderRight: "5px solid black",
            }}
          >
            <div style={{ height: "100vh" }}>
              <Camera paneWidth={cameraPaneWidth} />
            </div>
          </div>
          <div ref={mapDivRef} className="bg-slate-500">
            <div style={{ height: "100vh" }}>
              <Map socrataData={socrataData} paneWidth={mapPaneWidth} />
            </div>
          </div>
        </Allotment>
      </div>
    </>
  )
  // ...
}
