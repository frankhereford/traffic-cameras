import React, { useEffect, useState } from "react";
import type { DetectedObject, DetectionResult } from "./camera";

interface CameraBoundingBoxesProps {
  detections: DetectionResult;
}

export default function CameraBoundingBoxes({
  detections,
}: CameraBoundingBoxesProps) {
  const [boxes, setBoxes] = useState<JSX.Element[]>([]);

  useEffect(() => {
    console.log("detections: ", JSON.stringify(detections, null, 2));

    const newBoxes = detections.detected_objects.map((obj: DetectedObject) => {
      const { location, label } = obj;
      const [left, top, right, bottom] = location;
      const width = right! - left!;
      const height = bottom! - top!;

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
  }, [detections]);

  return <>{boxes}</>;
}
