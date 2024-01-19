import { useState, useEffect } from "react";
import Image from "next/image";
import useApplicationStore from "~/pages/hooks/applicationstore";

function Camera() {
  const camera = useApplicationStore((state) => state.camera);
  const [imageKey, setImageKey] = useState(Date.now());

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
          // onLoad={handleImageLoad}
        />
      )}
    </>
  );
}

export default Camera;
