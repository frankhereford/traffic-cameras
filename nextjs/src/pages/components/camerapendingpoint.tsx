import React, { useEffect, useState } from "react";
import useApplicationStore from "../hooks/applicationstore";
import styles from "./cameramarkers.module.css";

type CameraPoint = {
  x: number;
  y: number;
};

const markerSize = 5; // Half of the marker's size

export default function CameraPendingPoint() {
  const pendingCameraPoint = useApplicationStore(
    (state) => state.pendingCameraPoint,
  );

  const paneWidths = useApplicationStore((state) => state.paneWidths);

  const [cameraPoint, setCameraPoint] = useState<CameraPoint | null>(null);

  useEffect(() => {
    const img = document.getElementById("camera") as HTMLImageElement;
    const naturalWidth = img?.naturalWidth;
    const naturalHeight = img?.naturalHeight;
    const resizedWidth = img?.width;
    const resizedHeight = img?.height;

    if (
      pendingCameraPoint &&
      naturalWidth &&
      naturalHeight &&
      resizedWidth &&
      resizedHeight
    ) {
      const xRatio = resizedWidth / naturalWidth;
      const yRatio = resizedHeight / naturalHeight;

      const x = pendingCameraPoint.x * xRatio;
      const y = pendingCameraPoint.y * yRatio;

      setCameraPoint({ x, y });
    }
  }, [pendingCameraPoint, paneWidths]);
  return (
    <>
      {pendingCameraPoint && cameraPoint && (
        <div
          className={styles.marker}
          style={{
            left: `${cameraPoint.x - markerSize}px`,
            top: `${cameraPoint.y - markerSize}px`,
          }}
        />
      )}
    </>
  );
}
