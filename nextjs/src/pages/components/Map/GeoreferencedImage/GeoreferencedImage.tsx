/* eslint-disable @next/next/no-img-element */
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useState } from "react"
import { OverlayView } from "@react-google-maps/api"
import useTransformedImage from "~/pages/hooks/useTransformedImage"

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

export default function GeoreferencedImage({
  camera,
}: GeoreferencedImageProps) {
  const showTransformedImage = useTransformedImage(
    (state) => state.showTransformedImage,
  )

  const opacity = useTransformedImage((state) => state.opacity)

  const [bounds, setBounds] = useState<google.maps.LatLngBounds | null>(null)
  const [overlaySource, setOverlaySource] = useState<string | null>(null)
  const queryClient = useQueryClient()
  const { data: transformedImage } = useQuery<TransformedImage>(
    ["transformedImage", camera],
    () =>
      fetch(`/flask/transformedImage/${camera}`).then((response) =>
        response.json(),
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
      // console.log(JSON.stringify(transformedImage, null, 2))
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

  // useEffect(() => {
  //   console.log(`Opacity: ${opacity}`)
  // }, [opacity])

  return (
    <>
      {showTransformedImage && transformedImage && bounds && (
        <>
          <OverlayView mapPaneName={OverlayView.MAP_PANE} bounds={bounds}>
            <div
              style={{
                height: "100%",
                width: "100%",
                background: "#fff0",
                outline: "1px dashed #f00f", // my new favorite hex color
                opacity: opacity / 100, // convert opacity to range 0.0-1.0
              }}
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
