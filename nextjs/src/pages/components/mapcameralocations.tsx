import { useEffect, useState } from "react";
import useApplicationStore from "../hooks/applicationstore";
import { Marker } from "@react-google-maps/api";

import { api } from "~/utils/api";

export default function MapCameraLocations() {
  const cameraLocations = api.camera.getCameras.useQuery({});
  const allCameraData = useApplicationStore((state) => state.allCameraData);
  const setCamera = useApplicationStore((state) => state.setCamera);
  const [markers, setMarkers] = useState<(JSX.Element | undefined)[]>([]);

  useEffect(() => {
    if (cameraLocations.data && allCameraData) {
      const newMarkers = cameraLocations.data.map((camera) => {
        const matchingCamera = allCameraData.find(
          (allCamera) => allCamera.camera_id === camera.cameraId.toString(),
        );

        const status = camera?.status?.slug;

        if (matchingCamera) {
          const lat = matchingCamera.location.coordinates[1];
          const lng = matchingCamera.location.coordinates[0];

          if (typeof lat === "number" && typeof lng === "number") {
            let markerColor;

            switch (status) {
              case "ok":
                markerColor = "green-dot";
                break;
              case "unavailable":
                markerColor = "yellow-dot";
                break;
              case "404":
                markerColor = "red-dot";
                break;
              default:
                markerColor = "blue-dot";
                break;
            }

            return (
              <Marker
                key={camera.cameraId}
                position={{ lat, lng }}
                icon={{
                  url: `http://maps.google.com/mapfiles/ms/icons/${markerColor}.png`,
                }}
                onClick={() => {
                  setCamera(camera.cameraId);
                  console.log(
                    `Marker with cameraId ${camera.cameraId} was clicked.`,
                  );
                }}
              />
            );
          }
        }
      });

      setMarkers(
        newMarkers.filter((marker): marker is JSX.Element => Boolean(marker)),
      );
    }
  }, [cameraLocations.data, allCameraData]);

  return <>{markers}</>;
}
