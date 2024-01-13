import React, { useEffect, useState } from "react"
import { Check, ChevronsUpDown } from "lucide-react"
import { cn } from "~/utils"
import { Button } from "~/pages/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
} from "~/pages/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "~/pages/ui/popover"
import useIntersectionStore from "~/pages/hooks/IntersectionStore"

export interface Camera {
  camera_id: string
  location_name: string
  camera_status: string
  camera_mfg: string
  atd_location_id: string
  council_district: string
  jurisdiction_label: string
  location_type: string
  primary_st_block: string
  primary_st: string
  cross_st_block: string
  cross_st: string
  coa_intersection_id: string
  modified_date: string
  screenshot_address: string
  funding: string
  id: string
  location: {
    type: string
    coordinates: number[]
  }
}

const CameraPicker: React.FC = ({}) => {
  const [open, setOpen] = useState(false)
  const [value, setValue] = useState("")
  const [selectedCameraLocation, setSelectedCameraLocation] = useState("")
  const [cameraData, setCameraData] = useState<Camera[]>([])
  const camera = useIntersectionStore((state) => state.camera)
  const setCamera = useIntersectionStore((state) => state.setCamera)
  const setGlobalCameraData = useIntersectionStore(
    (state) => state.setCameraData,
  )

  useEffect(() => {
    fetch("https://data.austintexas.gov/resource/b4k4-adkb.json")
      .then((response) => response.json())
      .then((data: Camera[]) => {
        setCameraData(data)
        //console.log(JSON.stringify(data, null, 2)) // Pretty print the JSON string
      })
      .catch((error) => console.error("Error:", error))
  }, [])

  useEffect(() => {
    if (cameraData && camera) {
      const cameraObject = cameraData.find(
        (item) => parseInt(item.camera_id) === camera,
      )
      if (cameraObject) {
        // console.log("Found camera object:", cameraObject)
        setGlobalCameraData(cameraObject)
      } else {
        // console.log("No camera object found")
      }
    } else {
      // console.log("cameraData or camera is null")
    }
  }, [cameraData, camera])

  useEffect(() => {
    if (value !== "") {
      setCamera(parseInt(value))
      shuffleArray(cameraData)
      setCameraData([...cameraData])
    }
  }, [value])

  const shuffleArray = (array: Camera[]) => {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[array[i], array[j]] = [array[j]!, array[i]!]
    }
  }

  return (
    <>
      <div className="flex flex-col">
        <div>
          <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                role="combobox"
                aria-expanded={open}
                className="w-[600px] justify-between"
              >
                {selectedCameraLocation.toUpperCase() || "Select camera..."}
                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[600px] p-0">
              <Command>
                <CommandInput placeholder="Search camera..." />
                <CommandEmpty>No camera found.</CommandEmpty>
                <CommandGroup>
                  {cameraData.map((camera) => (
                    <CommandItem
                      key={camera.camera_id}
                      value={camera.location_name}
                      onSelect={(currentValue) => {
                        const selectedCamera = cameraData.find((camera) => {
                          return (
                            camera.location_name.toLowerCase().trim() ===
                            currentValue.toLowerCase().trim()
                          )
                        })
                        if (selectedCamera) {
                          setValue(selectedCamera.camera_id)
                        }
                        setSelectedCameraLocation(currentValue)
                        setOpen(false)
                      }}
                    >
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          selectedCameraLocation === camera.location_name
                            ? "opacity-100"
                            : "opacity-0",
                        )}
                      />
                      {camera.location_name}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </Command>
            </PopoverContent>
          </Popover>
        </div>
      </div>
    </>
  )
}

export default CameraPicker
