import Button from "@mui/material/Button"
import { signOut } from "next-auth/react"

export default function Logout() {
  return (
    <Button
      className="mb-4 p-0"
      variant="contained"
      style={{ fontSize: "35px" }}
      onClick={() => void signOut()}
    >
      ✌️
    </Button>
  )
}
