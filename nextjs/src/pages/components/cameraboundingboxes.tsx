import React, { useEffect, useState } from "react";
import type { DetectedObject, DetectionResult } from "./camera";
import useApplicationStore from "../hooks/applicationstore";

interface CameraBoundingBoxesProps {
  detections: DetectionResult;
}

export default function CameraBoundingBoxes({
  detections,
}: CameraBoundingBoxesProps) {
  const [boxes, setBoxes] = useState<JSX.Element[]>([]);
  const paneWidths = useApplicationStore((state) => state.paneWidths);

  useEffect(() => {
    // console.log("detections: ", JSON.stringify(detections, null, 2));

    const img = document.getElementById("camera") as HTMLImageElement;
    const naturalWidth = img?.naturalWidth;
    const naturalHeight = img?.naturalHeight;
    const resizedWidth = img?.width;
    const resizedHeight = img?.height;

    const xRatio = resizedWidth / naturalWidth;
    const yRatio = resizedHeight / naturalHeight;

    const newBoxes = detections.detected_objects.map((obj: DetectedObject) => {
      const { location, label } = obj;
      // eslint-disable-next-line prefer-const
      let [left, top, right, bottom] = location;
      const width = (right! - left!) * xRatio;
      const height = (bottom! - top!) * yRatio;
      left! *= xRatio;
      top! *= yRatio;

      return (
        <div
          key={`${left}-${top}`}
          style={{
            position: "absolute",
            left: `${left}px`,
            top: `${top}px`,
            width: `${width}px`,
            height: `${height}px`,
            border: `2px solid ${label === "car" ? "#0f0" : "grey"}`,
            color: "white",
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            padding: "2px",
            fontSize: "10px",
          }}
        >
          {label}
        </div>
      );
    });

    setBoxes(newBoxes);
  }, [detections, paneWidths]);

  return <>{boxes}</>;
}
