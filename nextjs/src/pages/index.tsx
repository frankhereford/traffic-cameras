import { signIn, signOut, useSession } from "next-auth/react";
import Head from "next/head";
import Link from "next/link";

import { api } from "~/utils/api";

export default function Home() {
  const hello = api.post.hello.useQuery({ text: "from tRPC" });

  return (
    <>
      <Head>
        <title>Create T3 App</title>
        <meta name="description" content="Generated by create-t3-app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div
        style={{
          background:
            "linear-gradient(38deg, rgba(4,0,62,1) 0%, rgba(6,43,17,1) 36%, rgba(0,31,37,1) 100%)",
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <main className="flex min-h-screen flex-col items-center justify-center">
          <div className="container flex flex-col items-center justify-center gap-12 px-4 py-16 ">
            <div className="flex flex-col items-center gap-2">
              <AuthShowcase />
            </div>
          </div>
        </main>
      </div>
    </>
  );

  function AuthShowcase() {
    const { data: sessionData } = useSession();

    const { data: secretMessage } = api.post.getSecretMessage.useQuery(
      undefined, // no input
      { enabled: sessionData?.user !== undefined },
    );

    return (
      <div className="flex flex-col items-center justify-center gap-4">
        <p className="text-center text-2xl text-white">
          {sessionData && <span>Logged in as {sessionData.user?.name}</span>}
          {secretMessage && <span> - {secretMessage}</span>}
        </p>
        <button
          className="rounded-full bg-white/10 px-10 py-3 font-semibold text-white no-underline transition hover:bg-white/20"
          onClick={sessionData ? () => void signOut() : () => void signIn()}
        >
          {sessionData ? "Sign out" : "Sign in"}
        </button>
      </div>
    );
  }
}
