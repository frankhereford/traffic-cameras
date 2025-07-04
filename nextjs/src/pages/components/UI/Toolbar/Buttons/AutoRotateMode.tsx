import { useEffect, useRef, useState, useCallback } from "react"
import Button from "@mui/material/Button"
import Tooltip from "@mui/material/Tooltip"
import useCameraStore from "~/pages/hooks/useCameraStore"
import useAutoRotateMode from "~/pages/hooks/useAutoRotateMode"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import useEmojiFavicon from "~/pages/hooks/useEmojiFavicon"
import { api } from "~/utils/api"
import useMapViewportStore from "~/stores/useMapViewportStore"
import useGetSocrataData from "~/pages/hooks/useSocrataData"
import type { SocrataData } from "~/pages/hooks/useSocrataData"
import useCameraViewHistoryStore from "~/stores/useCameraViewHistoryStore"

const MIN_INTERVAL_S = 3
const MAX_INTERVAL_S = 15
const DEFAULT_INTERVAL_S = 15

export default function AutoRotateMode() {
  
  const autoRotateMode = useAutoRotateMode(
    (state) => state.autoRotateMode,
  )
  const setAutoRotateMode = useAutoRotateMode(
    (state) => state.setAutoRotateMode,
  )
  const autoRotateIntervalS = useAutoRotateMode(
    (state) => state.autoRotateIntervalS,
  )

  const [isHovered, setIsHovered] = useState(false)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const setCamera = useCameraStore((state) => state.setCamera)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)
  const setEmoji = useEmojiFavicon((state) => state.setEmoji)

  const { data: workingCameras } = api.camera.getWorkingCameras.useQuery({})
  const { data: socrataCameras } = useGetSocrataData()
  const mapBounds = useMapViewportStore((state) => state.bounds)
  const { viewedCameraIds, addCameraToViewHistory } = useCameraViewHistoryStore()

  const pickRandomCamera = useCallback(() => {
    if (!workingCameras || workingCameras.length === 0 || !socrataCameras || isFocus) {
      return;
    }

    let cameraToSetId: number | null = null;

    if (mapBounds) {
      const socrataCameraMap = new Map<number, SocrataData>();
      socrataCameras.forEach(sc => {
        if (sc.camera_id) {
          socrataCameraMap.set(parseInt(sc.camera_id, 10), sc);
        }
      });

      const camerasInExtent = workingCameras.filter(dbCamera => {
        const socrataCam = socrataCameraMap.get(dbCamera.coaId);
        if (!socrataCam?.location?.coordinates) return false;
        
        const [longitude, latitude] = socrataCam.location.coordinates;
        if (typeof latitude !== 'number' || typeof longitude !== 'number') return false;

        const cameraPosition = new google.maps.LatLng(latitude, longitude);
        return mapBounds.contains(cameraPosition);
      }).map(cam => cam.coaId);

      if (camerasInExtent.length > 0) {
        const unviewedInExtent = camerasInExtent.filter(id => !viewedCameraIds.includes(id));
        if (unviewedInExtent.length > 0) {
          cameraToSetId = unviewedInExtent[0]!;
        } else {
          for (const historicId of viewedCameraIds) {
            if (camerasInExtent.includes(historicId)) {
              cameraToSetId = historicId;
              break;
            }
          }
          if (cameraToSetId === null) {
            cameraToSetId = camerasInExtent[Math.floor(Math.random() * camerasInExtent.length)]!;
          }
        }
      }
    }

    if (cameraToSetId === null) {
      const randomCamera = workingCameras[Math.floor(Math.random() * workingCameras.length)]!;
      cameraToSetId = randomCamera.coaId;
    }

    if (cameraToSetId !== null) {
      setCamera(cameraToSetId);
      addCameraToViewHistory(cameraToSetId);
      setEmoji("🔀");
    }
  }, [workingCameras, socrataCameras, isFocus, setCamera, setEmoji, mapBounds, viewedCameraIds, addCameraToViewHistory]);

  useEffect(() => {
    console.log("AutoRotateMode effect: ", autoRotateMode)
    if (autoRotateMode) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      intervalRef.current = setInterval(() => {
        pickRandomCamera()
      }, autoRotateIntervalS * 1000)
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [autoRotateMode, pickRandomCamera, autoRotateIntervalS])

  const handleClick = () => {
    setAutoRotateMode(!autoRotateMode)
  }

  if (!workingCameras || workingCameras.length === 0) return <></>

  return (
    <Tooltip title={`Toggle auto camera (${autoRotateIntervalS}s): least recent in map, else any random`}>
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px", position: "relative" }}
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        color={autoRotateMode ? "success" : "primary"}
      >
        {autoRotateMode ? "🔁" : "▶️"}
        {isHovered && (
          <span
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              fontSize: "50px",
              opacity: 0.15,
            }}
          >
            a
          </span>
        )}
      </Button>
    </Tooltip>
  )
}
