import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import { signOut } from "next-auth/react"

export default function Logout() {
  const [isHovered, setIsHovered] = useState(false)

  const handleSignOut = () => {
    signOut().catch((e) => {
      console.error(e)
    })
  }

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "l") {
        handleSignOut()
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [])

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
