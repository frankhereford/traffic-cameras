import { useState, useEffect } from "react";
import Image from "next/image";
import useApplicationStore from "~/pages/hooks/applicationstore";
import CryptoJS from "crypto-js";
import { api } from "~/utils/api";
import CameraPendingPoint from "./camerapendingpoint";
import CameraCorrelatedPoints from "./cameracorrelatedpoints";
import CameraBoundingBoxes from "./cameraboundingboxes";
export interface CameraPoint {
  x: number;
  y: number;
}
export interface CorrelatedPoint {
  id: string;
  cameraX: number;
  cameraY: number;
  mapLatitude: number;
  mapLongitude: number;
  cameraId: string;
  createdAt: Date;
  updatedAt: Date;
}

export type DetectedObject = {
  confidence: number;
  label: string;
  location: number[];
};

export type DetectionResult = {
  detected_objects: DetectedObject[];
};

function Camera() {
  const camera = useApplicationStore((state) => state.camera);
  const [imageKey, setImageKey] = useState(Date.now());

  const [cameraHex, setCameraHex] = useState<string | null>(null);
  const [cameraResponse, setCameraResponse] = useState<number | null>(null);
  const [base64Data, setBase64Data] = useState<string>("");
  const [detectionResult, setDetectionResult] =
    useState<DetectionResult | null>(null);

  const setPendingCameraPoint = useApplicationStore(
    (state) => state.setPendingCameraPoint,
  );

  const reload = useApplicationStore((state) => state.reload);

  const setStatus = api.camera.setStatus.useMutation({});

  //set up a query to get this camera's correlated points
  const correlatedPoints = api.correlatedPoints.getPointPairs.useQuery(
    {
      cameraId: camera!,
    },
    {
      enabled: !!camera,
    },
  );

  useEffect(() => {
    if (!cameraHex || !detectionResult) return;
    console.log("cameraHex:", cameraHex);
    console.log("detectionResult:", detectionResult);
  }, [detectionResult]);

  // log the outcome of the setStatus mutation
  useEffect(() => {
    if (setStatus.status === "success") {
      // The mutation has finished successfully
      // console.log("setStatus mutation finished successfully");
      // console.log(setStatus.data);
    } else if (setStatus.status === "error") {
      // The mutation has finished with an error
      console.log("setStatus mutation finished with an error");
    }
  }, [setStatus.status]);

  // refresh the camera image every 5 minutes
  useEffect(() => {
    const timer = setTimeout(
      () => {
        setImageKey(Date.now()); // Change key to force re-render
      },
      5 * 60 * 1000, // 5 minutes
      // 30 * 1000, // 30 seconds
    );

    return () => clearTimeout(timer); // Clear timeout if the component is unmounted
  }, [imageKey]);

  const url = `https://cctv.austinmobility.io/image/${camera}.jpg`;

  const handleImageLoad = (event: React.SyntheticEvent<HTMLImageElement>) => {
    const img = event.currentTarget;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    ctx?.drawImage(img, 0, 0);

    const base64data = canvas.toDataURL("image/jpeg");
    setBase64Data(base64data);
    const hash = CryptoJS.SHA256(base64data);
    const hex = hash.toString(CryptoJS.enc.Hex);
    // setDetectionResult(null);
    setCameraHex(hex);
  };

  // detect a new image's base64 data and get it processed by the object detection API
  useEffect(() => {
    setDetectionResult(null);
    if (base64Data) {
      console.log("asking for vision for hash: ", cameraHex);
      fetch("/flask/vision", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: base64Data }),
      })
        .then((response) => response.json())
        .then((data) => {
          // console.log("vision: ", JSON.stringify(data, null, 2));
          setDetectionResult(data as DetectionResult);
          // eslint-disable-next-line @typescript-eslint/no-unsafe-return
          return data;
        })
        .catch((error) => console.error("Error:", error));
    }
  }, [base64Data]);

  // save a status based on the hash of the image we got
  useEffect(() => {
    setCameraResponse(200);
    if (cameraHex) {
      if (
        cameraHex ===
        "6b7288a33808e35f205f33f8fdff8c7df822b0cf5595c99d86a7b9b6ca4238f9"
      ) {
        setStatus.mutate({
          cameraId: camera!,
          status: "unavailable",
          hex: cameraHex,
        });
      } else {
        setStatus.mutate({
          cameraId: camera!,
          status: "ok",
          hex: cameraHex,
        });
      }
    }
  }, [cameraHex]);

  //set a status if we got a 404
  useEffect(() => {
    if (cameraResponse === 404) {
      console.log("cameraResponse changed:", cameraResponse);
      setStatus.mutate({
        cameraId: camera!,
        status: "404",
      });
    }
  }, [cameraResponse]);

  const handleImageError = (event: React.SyntheticEvent<HTMLImageElement>) => {
    setCameraResponse(404);
  };

  const handleClick = async (
    event: React.MouseEvent<HTMLImageElement, MouseEvent>,
  ) => {
    const img = event.currentTarget;
    const rect = img.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    // console.log(`Clicked at coordinates: ${x}, ${y}`);
    const xRatio = img.naturalWidth / img.width;
    const yRatio = img.naturalHeight / img.height;
    const nativeX = Math.floor(x * xRatio);
    const nativeY = Math.floor(y * yRatio);
    // console.log(`Clicked at native coordinates: ${nativeX}, ${nativeY}`);
    const pendingCameraPoint = {
      x: nativeX,
      y: nativeY,
    };
    setPendingCameraPoint(pendingCameraPoint);
  };

  // this is garbage
  useEffect(() => {
    correlatedPoints
      .refetch()
      // .then(() => {
      //   console.log("refetched");
      // })
      .catch(console.error);
  }, [reload]);

  return (
    <>
      {camera && (
        <>
          <Image
            id="camera"
            key={imageKey}
            priority
            src={`${url}?${new Date().getTime()}`}
            width={1920}
            height={1080}
            alt="CCTV Camera"
            onClick={handleClick}
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
          <CameraPendingPoint />
          <CameraCorrelatedPoints
            points={correlatedPoints.data as CorrelatedPoint[]}
          />
          {detectionResult && (
            <CameraBoundingBoxes detections={detectionResult} />
          )}
        </>
      )}
    </>
  );
}

export default Camera;
