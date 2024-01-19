import { useEffect, useState } from "react";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import useApplicationStore from "~/pages/hooks/applicationstore";

export interface Camera {
  camera_id: string;
  location_name: string;
  camera_status: string;
  camera_mfg: string;
  atd_location_id: string;
  council_district: string;
  jurisdiction_label: string;
  location_type: string;
  primary_st_block: string;
  primary_st: string;
  cross_st_block: string;
  cross_st: string;
  coa_intersection_id: string;
  modified_date: string;
  screenshot_address: string;
  funding: string;
  id: string;
  location: {
    type: string;
    coordinates: number[];
  };
}

interface SimplifiedCamera {
  id: number;
  label: string;
}

function CameraPicker() {
  const [socrataDataSet, setSocrataDataSet] = useState<Camera[]>([]);
  const [cameraData, setCameraData] = useState<SimplifiedCamera[]>([]);
  const camera = useApplicationStore((state) => state.camera);
  const setCamera = useApplicationStore((state) => state.setCamera);
  const setCameraDetailsData = useApplicationStore(
    (state) => state.setCameraData,
  );

  // get cameraData from open data portal and default camera from local storage
  useEffect(() => {
    fetch("https://data.austintexas.gov/resource/b4k4-adkb.json")
      .then((response) => response.json())
      .then((cameraData: Camera[]) => {
        setSocrataDataSet(cameraData);
        return cameraData;
      })
      .then((cameraData: Camera[]) => {
        const autoCompleteData = cameraData
          .map((item) => {
            return {
              id: parseInt(item.camera_id),
              label: item.location_name,
            };
          })
          .reduce((unique: SimplifiedCamera[], o: SimplifiedCamera) => {
            if (!unique.some((obj) => obj.id === o.id)) {
              unique.push(o);
            }
            return unique;
          }, [])
          .reduce((unique: SimplifiedCamera[], o: SimplifiedCamera) => {
            if (
              !unique.some((obj: SimplifiedCamera) => obj.label === o.label)
            ) {
              unique.push(o);
            }
            return unique;
          }, []);
        setCameraData(autoCompleteData);
      })
      .catch((error) => console.error("Error:", error));

    // get cameraId from local storage
    const cameraId = parseInt(
      localStorage.getItem("cameraId") as unknown as string,
    );
    if (cameraId) {
      setCamera(cameraId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // store the selected camera in local storage
    if (camera) {
      localStorage.setItem("cameraId", camera.toString());
    }

    if (socrataDataSet) {
      const details = socrataDataSet.find(
        (item) => parseInt(item.camera_id) === camera,
      )!;
      if (details) {
        setCameraDetailsData(details);
      }
    }
  }, [camera, socrataDataSet]);

  return (
    <>
      {cameraData.length > 0 && (
        <Autocomplete
          disablePortal
          options={cameraData}
          filterOptions={(options, state) => {
            return options.filter((option) =>
              option.label
                .toLowerCase()
                .includes(state.inputValue.toLowerCase()),
            );
          }}
          value={camera ? cameraData.find((item) => item.id === camera) : null}
          sx={{ width: 200 }}
          className="mb-2"
          renderInput={(params) => <TextField {...params} label="Camera" />}
          onChange={(event, newValue) => {
            if (newValue?.id) {
              setCamera(newValue.id);
            }
          }}
        />
      )}
    </>
  );
}

export default CameraPicker;
