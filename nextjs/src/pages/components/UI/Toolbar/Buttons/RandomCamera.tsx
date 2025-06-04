import { useEffect, useState, useCallback } from "react"
import Button from "@mui/material/Button"
import useCameraStore from "~/pages/hooks/useCameraStore"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import Tooltip from "@mui/material/Tooltip"
import useEmojiFavicon from "~/pages/hooks/useEmojiFavicon"

import { api } from "~/utils/api"
import useMapViewportStore from "~/stores/useMapViewportStore"
import useGetSocrataData from "~/pages/hooks/useSocrataData"
import type { SocrataData } from "~/pages/hooks/useSocrataData"
import useCameraViewHistoryStore from "~/stores/useCameraViewHistoryStore"

export default function RandomCamera() {
  const setCamera = useCameraStore((state) => state.setCamera)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)
  const setEmoji = useEmojiFavicon((state) => state.setEmoji)
  const mapBounds = useMapViewportStore((state) => state.bounds)
  const { viewedCameraIds, addCameraToViewHistory } = useCameraViewHistoryStore()

  // data from getWorkingCameras (cameras with status 'ok')
  const { data: workingCameras, isLoading: isLoadingWorkingCameras } =
    api.camera.getWorkingCameras.useQuery({})

  // data from Socrata (cameras with locations and 'TURNED_ON' status)
  const { data: socrataCameras, isLoading: isLoadingSocrata } =
    useGetSocrataData()

  const [isHovered, setIsHovered] = useState(false)

  const handleClick = useCallback(
    (
      event?: React.MouseEvent<HTMLButtonElement>,
      options: { filterByExtent?: boolean; filterByGeoreference?: boolean } = {},
    ) => {
      if (workingCameras && socrataCameras && !isFocus) {
        let camerasToConsider = workingCameras;
        const { filterByExtent = false, filterByGeoreference = false } = options;

        if (filterByGeoreference) {
          camerasToConsider = camerasToConsider.filter(
            (item) => item._count.Location >= 5,
          );
        }

        // Socrata data mapping for location lookup
        const socrataCameraMap = new Map<number, SocrataData>();
        socrataCameras.forEach(sc => {
          if (sc.camera_id) {
            socrataCameraMap.set(parseInt(sc.camera_id, 10), sc);
          }
        });

        let finalFilteredCameras = camerasToConsider.filter(dbCamera => {
          const socrataCam = socrataCameraMap.get(dbCamera.coaId);
          // If filtering by extent, camera must have location data
          if (filterByExtent && !socrataCam?.location?.coordinates) {
            return false;
          }
          // If not filtering by extent, but it has no socrataCam entry, it can still be included if not georeferenced
          // However, if we ARE filtering by extent, it MUST have socrataCam and coords.

          if (filterByExtent && mapBounds && socrataCam?.location?.coordinates) {
            const [longitude, latitude] = socrataCam.location.coordinates;
            if (typeof latitude !== 'number' || typeof longitude !== 'number') {
              return false; 
            }
            const cameraPosition = new google.maps.LatLng(latitude, longitude);
            if (!mapBounds.contains(cameraPosition)) {
              return false; // Outside map bounds
            }
          }
          return true; // Passes all applicable filters
        });

        if (finalFilteredCameras.length > 0) {
          let cameraToSetId: number | null = null;

          if (filterByExtent && !filterByGeoreference) { // This is the Shift+R case
            const candidatesInExtent = finalFilteredCameras.map(cam => cam.coaId);
            
            // 1. Find cameras in extent that have *not* been viewed
            const unviewedInExtent = candidatesInExtent.filter(id => !viewedCameraIds.includes(id));
            if (unviewedInExtent.length > 0) {
              cameraToSetId = unviewedInExtent[0]!; // Pick the first unviewed one
            } else {
              // 2. If all in extent have been viewed, find the least recently viewed among them
              // Iterate through history (oldest to newest)
              for (const historicId of viewedCameraIds) {
                if (candidatesInExtent.includes(historicId)) {
                  cameraToSetId = historicId;
                  break;
                }
              }
              // If somehow cameraToSetId is still null (e.g. candidatesInExtent was empty but passed initial check - unlikely)
              // Fallback to random from this set, though ideally this path isn't hit if candidatesInExtent had items.
              if (cameraToSetId === null && candidatesInExtent.length > 0) {
                 cameraToSetId = candidatesInExtent[Math.floor(Math.random() * candidatesInExtent.length)]!;
              }
            }
          } else {
            // Default: pick randomly from the filtered list
            const randomCamera =
              finalFilteredCameras[Math.floor(Math.random() * finalFilteredCameras.length)]!;
            cameraToSetId = randomCamera.coaId;
          }

          if (cameraToSetId !== null) {
            setCamera(cameraToSetId);
            addCameraToViewHistory(cameraToSetId); // Add to history
            setEmoji("🎯");
          } else {
            console.log("No suitable camera found to set after applying all filters.");
          }
        } else {
          console.log("No cameras found matching the current criteria.");
        }
      }
    },
    [workingCameras, socrataCameras, isFocus, mapBounds, setCamera, setEmoji, viewedCameraIds, addCameraToViewHistory] // Dependencies for useCallback
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();

      if (key === 'r') {
        if (event.ctrlKey || event.metaKey) { // Ctrl+R or Cmd+R
          event.preventDefault();
          handleClick(undefined, { filterByExtent: false, filterByGeoreference: true });
        } else if (event.shiftKey) { // Shift+R
          event.preventDefault();
          handleClick(undefined, { filterByExtent: false, filterByGeoreference: false });
        } else if (!event.altKey) { // Plain 'r'
          // No event.preventDefault() needed for plain 'r' normally,
          // but if it triggers map extent filtering, consider if default browser 'r' actions interfere.
          // For now, let's assume it's fine or handleClick itself doesn't cause page reloads.
          handleClick(undefined, { filterByExtent: true, filterByGeoreference: false });
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleClick]); // handleClick is now stable due to useCallback

  // Adjust loading state check
  const isLoading = isLoadingWorkingCameras || isLoadingSocrata;

  return (
    <>
      {!isLoading && workingCameras && workingCameras.length > 0 && (
        <Tooltip title="Random Camera | Click: in map (least recent), Shift+Click: any | Keys: [r] in map (LRU), [Shift+r] any, [Ctrl+r] georef">
          <Button
            className="mb-4 p-0"
            variant="contained"
            style={{ fontSize: "35px", position: "relative" }}
            onClick={(event) => {
              if (event.shiftKey) {
                // Shift+Click: random camera anywhere (same as Shift+R)
                handleClick(event, { filterByExtent: false, filterByGeoreference: false });
              } else {
                // Normal Click: least recent in map extent (same as plain 'r')
                handleClick(event, { filterByExtent: true, filterByGeoreference: false });
              }
            }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
          >
            🎯
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
                r
              </span>
            )}
          </Button>
        </Tooltip>
      )}
    </>
  )
}
