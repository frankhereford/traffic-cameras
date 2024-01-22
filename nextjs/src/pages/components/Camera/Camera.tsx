interface CameraProps {
  paneWidth: number
}

export default function Camera({ paneWidth }: CameraProps) {
  return (
    <>
      <div>{paneWidth}</div>
    </>
  )
}
