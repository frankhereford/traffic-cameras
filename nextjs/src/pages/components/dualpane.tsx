import { useSession } from "next-auth/react";
import { Allotment } from "allotment";
import "allotment/dist/style.css";
import Auth from "./auth";

export default function DualPane() {
  const { data: sessionData } = useSession();

  console.log(sessionData);

  return (
    <>
      <div style={{ height: "100vh", width: "100vw" }}>
        <Allotment defaultSizes={[100, 200]}>
          <div className="bg-indigo-500">
            <div style={{ height: "100vh" }}>
              <Auth />
            </div>
          </div>
          <div className="bg-green-500">
            <div style={{ height: "100vh" }}>
              <Auth />
            </div>
          </div>
        </Allotment>
      </div>
    </>
  );
}
