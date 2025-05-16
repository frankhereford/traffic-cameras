import "@fontsource/roboto/300.css"
import "@fontsource/roboto/400.css"
import "@fontsource/roboto/500.css"
import "@fontsource/roboto/700.css"

import Head from "next/head"
import CameraGeoreferenceApp from "~/pages/components/Application/CameraGeoreferenceApp"
import useEmojiFavicon from "~/pages/hooks/useEmojiFavicon"

export default function Home() {
  const emoji = useEmojiFavicon((state) => state.emoji)

  return (
    <>
      <Head>
        <title>Traffic Camera Georeferencer</title>
        <meta name="description" content="Traffic Camera Georeferencer" />
        <link
          rel="icon"
          href={`data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='0.9em' font-size='90'>${emoji}</text></svg>`}
        />
      </Head>
      <main>
        <CameraGeoreferenceApp />
      </main>
    </>
  )
}
