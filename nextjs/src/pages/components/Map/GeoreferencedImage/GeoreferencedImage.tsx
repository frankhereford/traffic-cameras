/* eslint-disable @next/next/no-img-element */
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useState } from "react"
import { OverlayView } from "@react-google-maps/api"

interface GeoreferencedImageProps {
  camera: number
}

interface GeojsonFeature {
  type: string
  properties: unknown
  geometry: {
    type: string
    coordinates: number[][]
  }
}

interface TransformedImage {
  extent: number[]
  geojson: {
    type: string
    features: GeojsonFeature[]
  }
  image: string
}

interface OverlayViewStyle {
  background: string
  width: string
  height: string
}

export default function GeoreferencedImage({
  camera,
}: GeoreferencedImageProps) {
  const [bounds, setBounds] = useState<google.maps.LatLngBounds | null>(null)
  const [overlaySource, setOverlaySource] = useState<string | null>(null)
  const queryClient = useQueryClient()
  const { data: transformedImage, refetch } = useQuery<TransformedImage>(
    ["transformedImage", camera],
    () =>
      fetch(`http://localhost/flask/transformedImage/${camera}`).then(
        (response) => response.json(),
      ),
  )

  useEffect(() => {
    queryClient
      .invalidateQueries([["transformedImage", camera]])
      .catch((error) => {
        console.log("error: ", error)
      })
  }, [camera, queryClient])

  useEffect(() => {
    if (transformedImage) {
      console.log(JSON.stringify(transformedImage, null, 2))
      setBounds(
        new google.maps.LatLngBounds(
          {
            lat: transformedImage.extent[1]!,
            lng: transformedImage.extent[0]!,
          }, // southwest corner
          {
            lat: transformedImage.extent[3]!,
            lng: transformedImage.extent[2]!,
          }, // northeast corner
        ),
      )
      setOverlaySource(`data:image/png;base64,${transformedImage.image}`)
    }
  }, [transformedImage])

  const getPixelPositionOffset = (width: number, height: number) => ({
    x: -(width / 2),
    y: -(height / 2),
  })

  return (
    <>
      {transformedImage && bounds && (
        <>
          <OverlayView
            mapPaneName={OverlayView.OVERLAY_MOUSE_TARGET}
            bounds={bounds}
            // getPixelPositionOffset={getPixelPositionOffset}
          >
            <div
              style={{ height: "100%", width: "100%", background: "#ffffff22" }}
            >
              <img
                src={overlaySource!}
                alt="Georeferenced Image"
                style={{ objectFit: "fill", height: "100%", width: "100%" }}
              />{" "}
            </div>
          </OverlayView>
        </>
      )}
    </>
  )
}
