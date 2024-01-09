import React, { useEffect } from 'react';


interface CCTVImageProps {
    cameraId: number;
    coordinates: Array<{ cctv: [number, number], map: [number, number] }>;
    createPoint: (x: number, y: number) => { x: number, y: number };
}

const CCTVImage: React.FC<CCTVImageProps> = ({ cameraId, coordinates }) => {
    const cctv = 'https://cctv.austinmobility.io/image/' + cameraId + '.jpg';


    useEffect(() => {
        console.log(coordinates);
    }, [coordinates]);



    return (
        <img src={cctv} style={{ width: '50vw', height: '100vh', objectFit: 'contain' }} />
    );
};

export default CCTVImage;
