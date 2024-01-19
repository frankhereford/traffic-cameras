import { signIn, signOut, useSession } from "next-auth/react";
import { Button } from "@mui/material";
import { useTheme } from "@mui/material/styles";

function Auth() {
  const { data: sessionData } = useSession();
  const theme = useTheme();

  return (
    <div className="flex flex-col items-center justify-center gap-4">
      <Button
        variant="contained"
        style={{ background: theme.palette.primary.main }}
        onClick={sessionData ? () => void signOut() : () => void signIn()}
      >
        {sessionData ? "Sign out" : "Sign in"}
      </Button>
    </div>
  );
}

export default Auth;
