import { createTheme } from "@mui/material/styles";
// import { green, purple } from "@mui/material/colors";

export const theme = createTheme({
  components: {
    MuiAutocomplete: {
      defaultProps: {
        size: "small",
      },
    },
    MuiButton: {
      defaultProps: {
        size: "small",
        color: "success",
      },
    },
  },
});
