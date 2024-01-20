import { useState, useEffect } from "react";
import Image from "next/image";
import useApplicationStore from "~/pages/hooks/applicationstore";
import CryptoJS from "crypto-js";
import { api } from "~/utils/api";

function Camera() {
  const camera = useApplicationStore((state) => state.camera);
  const [imageKey, setImageKey] = useState(Date.now());
  const [cameraHex, setCameraHex] = useState<string | null>(null);
  const [cameraResponse, setCameraResponse] = useState<number | null>(null);
  const setStatus = api.camera.setStatus.useMutation({});

  useEffect(() => {
    const timer = setTimeout(
      () => {
        setImageKey(Date.now()); // Change key to force re-render
      },
      5 * 60 * 1000,
    ); // 5 minutes

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
    const hash = CryptoJS.SHA256(base64data);
    const hex = hash.toString(CryptoJS.enc.Hex);
    setCameraHex(hex);
  };

  useEffect(() => {
    if (cameraHex) {
      // console.log("cameraHex changed:", cameraHex);
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
    // console.log("handleImageError", event);
    setCameraResponse(404);
  };

  return (
    <>
      {camera && (
        <Image
          key={imageKey}
          priority
          src={`${url}?${new Date().getTime()}`}
          width={1920}
          height={1080}
          alt="CCTV Camera"
          // onClick={handleClick}
          onLoad={handleImageLoad}
          onError={handleImageError}
        />
      )}
    </>
  );
}

export default Camera;
