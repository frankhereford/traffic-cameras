import { useEffect, useState } from "react"
import Button from "@mui/material/Button"
import { signIn } from "next-auth/react"
import useAutocompleteFocus from "~/pages/hooks/useAutocompleteFocus"
import Tooltip from "@mui/material/Tooltip"
import useEmojiFavicon from "~/pages/hooks/useEmojiFavicon"

export default function Login() {
  const [isHovered, setIsHovered] = useState(false)
  const isFocus = useAutocompleteFocus((state) => state.isFocus)
  const setEmoji = useEmojiFavicon((state) => state.setEmoji)

  const handleSignIn = () => {
    setEmoji("🔑")
    signIn().catch((e) => {
      console.error(e)
    })
  }

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isFocus && event.key === "i") {
        handleSignIn()
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [isFocus])

  return (
    <Tooltip title="Sign in">
      <Button
        className="mb-4 p-0"
        variant="contained"
        style={{ fontSize: "35px", position: "relative" }}
        onClick={handleSignIn}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        🔑
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
            i
          </span>
        )}
      </Button>
    </Tooltip>
  )
}
