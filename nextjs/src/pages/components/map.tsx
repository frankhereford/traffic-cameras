import React, { useState, useEffect } from "react";
import { GoogleMap, useJsApiLoader } from "@react-google-maps/api";
import useApplicationStore from "~/pages/hooks/applicationstore";
import MapPendingMarker from "./mappendingmaker";
import { set } from "zod";

const containerStyle = {
  width: "100%",
  height: "100%",
};

function Map() {
  const { isLoaded } = useJsApiLoader({
    id: "google-map-script",
    googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAP_API_KEY ?? "",
  });

  const [map, setMap] = React.useState(null);
  const [center, setCenter] = useState<google.maps.LatLng | null>(null);
  const cameraData = useApplicationStore((state) => state.cameraData);
  const setPendingMapPoint = useApplicationStore(
    (state) => state.setPendingMapPoint,
  );

  // set the map center to the camera location
  useEffect(() => {
    if (isLoaded && cameraData?.location?.coordinates) {
      const [longitude, latitude] = cameraData.location.coordinates;
      setCenter(new google.maps.LatLng(latitude!, longitude));
    }
  }, [cameraData, isLoaded]);

  const onUnmount = React.useCallback(function callback() {
    setMap(null);
  }, []);

  const handleClick = (event: google.maps.MapMouseEvent) => {
    const lat = event.latLng?.lat();
    const lng = event.latLng?.lng();
    // console.log(`Clicked at ${lat}, ${lng}`);
    setPendingMapPoint(new google.maps.LatLng(lat!, lng));
  };

  return isLoaded ? (
    <GoogleMap
      mapContainerStyle={containerStyle}
      center={center ?? new google.maps.LatLng(30.262531, -97.753983)}
      zoom={17}
      onUnmount={onUnmount}
      options={{ tilt: 0, mapTypeId: "satellite" }}
      onClick={handleClick}
    >
      <MapPendingMarker />
    </GoogleMap>
  ) : (
    <></>
  );
}

export default React.memo(Map);
