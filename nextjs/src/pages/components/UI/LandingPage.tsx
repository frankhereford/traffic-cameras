import { signIn, signOut, useSession } from "next-auth/react"
import { Button } from "@mui/material"
import { useTheme } from "@mui/material/styles"

export default function LandingPage() {
  const { data: sessionData } = useSession()
  const theme = useTheme()

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(to right, #1f4037, #99f2c8)",
      }}
    >
      <Button
        variant="contained"
        style={{
          background: theme.palette.primary.main,
          padding: "20px 50px",
          fontSize: "20px",
        }}
        onClick={sessionData ? () => void signOut() : () => void signIn()}
      >
        {sessionData ? "Sign out" : "Sign in"}
      </Button>
    </div>
  )
}
