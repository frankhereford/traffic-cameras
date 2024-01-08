/* eslint-disable jsx-a11y/alt-text */
/* eslint-disable @next/next/no-img-element */
import Head from "next/head";
import Link from "next/link";
import Map from "~/pages/components/Map";

import { api } from "~/utils/api";

export default function Home() {

  const containerStyle = {
    width: '50vw', // 50% of viewport width
    height: '100vh' // 100% of viewport height
  };

  const center = {
    lat: 30.2672,
    lng: -97.7431
  };



  return (
    <>
      <Head>
        <title>Create T3 App</title>
        <meta name="description" content="Generated by create-t3-app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main style={{ display: 'flex', backgroundColor: '#333' }}> {/* Use Flexbox to split the main area and set the background color to dark grey */}
        <img src="https://cctv.austinmobility.io/image/326.jpg" style={{ width: '50vw', height: '100vh', objectFit: 'contain' }} /> {/* Scale the image to fit the horizontal space */}
        <Map center={center} containerStyle={containerStyle} />
      </main>
    </>
  );
}