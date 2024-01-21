import { useEffect, useState } from "react";

import { Marker } from "@react-google-maps/api";

export interface Point {
  id: string;
  cameraX: number;
  cameraY: number;
  mapLatitude: number;
  mapLongitude: number;
  cameraId: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface MapCorrelatedPointsProps {
  points: Point[];
}

export default function MapCorrelatedPoints({
  points,
}: MapCorrelatedPointsProps) {
  const [markers, setMarkers] = useState<JSX.Element[]>([]);

  useEffect(() => {
    if (points && points.length > 0) {
      const newMarkers = points.map((point) => (
        <Marker
          key={point.id}
          position={{ lat: point.mapLatitude, lng: point.mapLongitude }}
          icon={{
            url: `http://maps.google.com/mapfiles/ms/icons/blue.png`,
          }}
        />
      ));
      setMarkers(newMarkers);
    }
  }, [points]);

  return <>{markers}</>;
}
