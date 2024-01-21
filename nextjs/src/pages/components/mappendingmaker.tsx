import useApplicationStore from "../hooks/applicationstore";
import { Marker } from "@react-google-maps/api";

export default function MapPendingMarker() {
  const pendingMapPoint = useApplicationStore((state) => state.pendingMapPoint);

  // console.log("MapPendingMarker", pendingMapPoint);
  return (
    <>
      {pendingMapPoint && (
        <Marker
          position={pendingMapPoint}
          icon={{
            url: "http://maps.google.com/mapfiles/ms/icons/red-dot.png",
          }}
        />
      )}
    </>
  );
}
