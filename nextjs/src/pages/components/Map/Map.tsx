interface MapProps {
  paneWidth: number
}

export default function Map({ paneWidth }: MapProps) {
  return (
    <>
      <div>{paneWidth}</div>
    </>
  )
}
