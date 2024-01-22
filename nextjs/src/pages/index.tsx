import "@fontsource/roboto/300.css"
import "@fontsource/roboto/400.css"
import "@fontsource/roboto/500.css"
import "@fontsource/roboto/700.css"

import Head from "next/head"
import CameraGeoreferenceApp from "~/pages/components/Application/CameraGeoreferenceApp"

export default function Home() {
  return (
    <>
      <Head>
        <title>Traffic Camera Georeferencer</title>
        <meta name="description" content="Traffic Camera Georeferencer" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main>
        <CameraGeoreferenceApp />
      </main>
    </>
  )
}
