/* eslint-disable @typescript-eslint/no-unsafe-return */
/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
import React, { useEffect } from 'react';
import useCoordinateStore from '../hooks/CoordinateStore';

interface CCTVImageProps {
    cameraId: number;
}

const CCTVImage: React.FC<CCTVImageProps> = ({ cameraId }) => {
    const cctv = 'https://cctv.austinmobility.io/image/' + cameraId + '.jpg';

    const coordinates = useCoordinateStore((state) => state.coordinates);
    console.log("coordinates", coordinates);
    const addCoordinates = useCoordinateStore((state) => state.addCoordinates)


    return (
        <img src={cctv} style={{ width: '50vw', height: '100vh', objectFit: 'contain' }} />
    );
};

export default CCTVImage;
