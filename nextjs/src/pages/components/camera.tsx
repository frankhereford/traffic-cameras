import useApplicationStore from "~/pages/hooks/applicationstore";

function Camera() {
  const camera = useApplicationStore((state) => state.camera);
  return (
    <>
      <div>{camera}</div>
    </>
  );
}

export default Camera;
