import Head from "next/head"
import DualPane from "./components/DualPane"

// import { api } from "~/utils/api";

export default function Home() {
  return (
    <>
      <Head>
        <title>Traffic Camera Georeferencer</title>
        <meta name="description" content="Traffic Camera Georeferencer" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main>
        <div>
          <DualPane />
        </div>
      </main>
    </>
  )
}
