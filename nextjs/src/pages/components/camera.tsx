import Image from "next/image";
import useApplicationStore from "~/pages/hooks/applicationstore";

function Camera() {
  const camera = useApplicationStore((state) => state.camera);
  const url = `https://cctv.austinmobility.io/image/${camera}.jpg`;
  return (
    <>
      {camera && (
        <Image
          key={camera}
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
