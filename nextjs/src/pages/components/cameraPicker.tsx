import { Button } from "~/pages/ui/button"
import { ComboboxDemo } from "~/pages/ui/combobox"
import React, { useEffect } from "react"

const CameraPicker: React.FC = ({}) => {
  useEffect(() => {
    fetch("https://data.austintexas.gov/resource/b4k4-adkb.json")
      .then((response) => response.json())
      //.then((data) => setData(data[0]))
      .then((data) => console.log(data))
      .catch((error) => console.error("Error:", error))
  }, [])

  return (
    <>
      <div className="flex h-screen items-center justify-center">
        <Button>CameraPicker</Button>
        <ComboboxDemo />
      </div>
    </>
  )
}

export default CameraPicker
