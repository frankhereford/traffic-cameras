import { useEffect, useState } from "react";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";

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
  const [cameraData, setCameraData] = useState<SimplifiedCamera[]>([]);

  useEffect(() => {
    fetch("https://data.austintexas.gov/resource/b4k4-adkb.json")
      .then((response) => response.json())
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
  }, []);

  return (
    <>
      <Autocomplete
        disablePortal
        id="combo-box-demo"
        options={cameraData}
        filterOptions={(options, state) => {
          return options.filter((option) =>
            option.label.toLowerCase().includes(state.inputValue.toLowerCase()),
          );
        }}
        sx={{ width: 200 }}
        className="mb-2"
        renderInput={(params) => <TextField {...params} label="Camera" />}
      />
    </>
  );
}

export default CameraPicker;
