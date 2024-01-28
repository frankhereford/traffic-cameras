import { ThemeProvider, createTheme } from "@mui/material/styles"
import Box from "@mui/material/Box"
import LinearProgress from "@mui/material/LinearProgress"

type Props = {
  progress: number
}

export default function ReloadProgress({ progress }: Props) {
  const theme = createTheme({
    components: {
      MuiLinearProgress: {
        styleOverrides: {
          colorPrimary: {
            backgroundColor: "#00000000",
          },
          barColorPrimary: {
            backgroundColor: "light blue",
          },
        },
      },
    },
  })

  return (
    <Box sx={{ width: "100%", maxWidth: "1920px", marginTop: "-2px" }}>
      <ThemeProvider theme={theme}>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{ height: "2px" }}
        />
      </ThemeProvider>
    </Box>
  )
}
