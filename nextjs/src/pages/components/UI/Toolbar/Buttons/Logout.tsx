import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import { signOut } from "next-auth/react"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"

export default function Logout() {
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)

  const handleSignOut = () => {
    signOut().catch((e) => {
      console.error(e)
    })
  }

  useEffect(() => {
    console.log("isFocus:", isFocus)
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isFocus && event.key === "l") {
        handleSignOut()
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [isFocus])

  return (
    <Button
      className="mb-4 p-0"
      variant="contained"
      style={{ fontSize: "35px", position: "relative" }}
      onClick={handleSignOut}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      ✌️
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
          l
        </span>
      )}
    </Button>
  )
}
