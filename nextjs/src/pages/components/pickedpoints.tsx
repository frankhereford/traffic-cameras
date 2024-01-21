import { useEffect } from "react";
import useApplicationStore from "~/pages/hooks/applicationstore";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import { useTheme } from "@mui/material/styles";
import { api } from "~/utils/api";

export default function PickedPoints() {
  const pendingCameraPoint = useApplicationStore(
    (state) => state.pendingCameraPoint,
  );

  const camera = useApplicationStore((state) => state.camera);
  const pendingMapPoint = useApplicationStore((state) => state.pendingMapPoint);
  const theme = useTheme();

  useEffect(() => {
    console.log(pendingMapPoint);
  }, [pendingMapPoint]);

  const setCorrelatedPoint = api.correlatedPoints.setPointPair.useMutation({});

  const savePointPair = () => {
    console.log("Camera Point:", pendingCameraPoint);
    console.log("Map Point:", pendingMapPoint);
    if (camera && pendingCameraPoint && pendingMapPoint) {
      setCorrelatedPoint.mutate({
        cameraId: camera,
        cameraX: pendingCameraPoint.x,
        cameraY: pendingCameraPoint.y,
        mapLat: pendingMapPoint.lat(),
        mapLng: pendingMapPoint.lng(),
      });
    }
  };

  return (
    <div className="mb-2 mr-[20px] flex  text-black">
      <div className="flex-grow">
        <Typography>
          Camera:
          {pendingCameraPoint && (
            <span className="ml-2">
              ({pendingCameraPoint.x.toFixed(0)},{" "}
              {pendingCameraPoint.y.toFixed(0)})
            </span>
          )}
        </Typography>
      </div>
      <div className="flex-grow">
        <Typography>
          Map:
          {pendingMapPoint && (
            <span className="ml-2">
              ({pendingMapPoint.lat().toFixed(4)},{" "}
              {pendingMapPoint.lng().toFixed(4)})
            </span>
          )}
        </Typography>
      </div>
      <div className="w-[55px]">
        <Button
          className="ml-[-10px] mt-[-4px]"
          variant="contained"
          style={
            !(pendingCameraPoint && pendingMapPoint)
              ? {}
              : { background: theme.palette.primary.main }
          }
          onClick={savePointPair}
          disabled={!pendingCameraPoint || !pendingMapPoint}
        >
          Save
        </Button>
      </div>
    </div>
  );
}
