/* eslint-disable @typescript-eslint/no-unsafe-argument */
import React, { useEffect } from "react"
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

const frameworks = [
  {
    value: "next.js",
    label: "Next.js",
  },
  {
    value: "sveltekit",
    label: "SvelteKit",
  },
  {
    value: "nuxt.js",
    label: "Nuxt.js",
  },
  {
    value: "remix",
    label: "Remix",
  },
  {
    value: "astro",
    label: "Astro",
  },
]

const CameraPicker: React.FC = ({}) => {
  const [open, setOpen] = React.useState(false)
  const [value, setValue] = React.useState("")
  const [cameraData, setCameraData] = React.useState({})

  useEffect(() => {
    fetch("https://data.austintexas.gov/resource/b4k4-adkb.json")
      .then((response) => response.json())
      .then((data) => setCameraData(data))
      .then(() => console.log(cameraData))
      .catch((error) => console.error("Error:", error))
  }, [])

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-[200px] justify-between"
          >
            {value
              ? frameworks.find((framework) => framework.value === value)?.label
              : "Select framework..."}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[200px] p-0">
          <Command>
            <CommandInput placeholder="Search framework..." />
            <CommandEmpty>No framework found.</CommandEmpty>
            <CommandGroup>
              {frameworks.map((framework) => (
                <CommandItem
                  key={framework.value}
                  value={framework.value}
                  onSelect={(currentValue) => {
                    setValue(currentValue === value ? "" : currentValue)
                    setOpen(false)
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      value === framework.value ? "opacity-100" : "opacity-0",
                    )}
                  />
                  {framework.label}
                </CommandItem>
              ))}
            </CommandGroup>
          </Command>
        </PopoverContent>
      </Popover>
    </>
  )
}

export default CameraPicker
