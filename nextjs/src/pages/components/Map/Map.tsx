import type { SocrataData } from "~/pages/hooks/useSocrataData"

interface MapProps {
  paneWidth: number
  socrataData: SocrataData[]
}

export default function Map({ socrataData, paneWidth }: MapProps) {
  return (
    <>
      <div>{paneWidth}</div>
    </>
  )
}
