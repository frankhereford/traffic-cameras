import { useEffect, useState } from "react";
import styles from "./cameramarkers.module.css";
import useApplicationStore from "../hooks/applicationstore";

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

export interface CameraCorrelatedPointsProps {
  points: Point[];
}

const markerSize = 5;

export default function CameraCorrelatedPoints({
  points,
}: CameraCorrelatedPointsProps) {
  const [markers, setMarkers] = useState<JSX.Element[]>([]);
  const paneWidths = useApplicationStore((state) => state.paneWidths);

  useEffect(() => {
    if (!points || points.length === 0) {
      return;
    }
    // console.log("points!: ", points);
    // console.log(JSON.stringify(points, null, 2));

    const img = document.getElementById("camera") as HTMLImageElement;
    const naturalWidth = img?.naturalWidth;
    const naturalHeight = img?.naturalHeight;
    const resizedWidth = img?.width;
    const resizedHeight = img?.height;
    // console.log("naturalWidth: ", naturalWidth);
    // console.log("naturalHeight: ", naturalHeight);
    // console.log("resizedWidth: ", resizedWidth);
    // console.log("resizedHeight: ", resizedHeight);

    const markers = points.map((point) => {
      const xRatio = resizedWidth / naturalWidth;
      const yRatio = resizedHeight / naturalHeight;

      const x = point.cameraX * xRatio;
      const y = point.cameraY * yRatio;

      return (
        <>
          <div
            key={point.id}
            className={styles.pointpairmarker}
            style={{
              left: `${x - markerSize}px`,
              top: `${y - markerSize}px`,
            }}
          />
        </>
      );
    });
    setMarkers(markers);
  }, [points, paneWidths]);

  return <>{markers}</>;
}
