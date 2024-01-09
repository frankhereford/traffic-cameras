interface CCTVImageProps {
    cameraId: number;
}

const CCTVImage: React.FC<CCTVImageProps> = ({ cameraId }) => {
    const cctv = 'https://cctv.austinmobility.io/image/' + cameraId + '.jpg';
    return (
        <img src={cctv} style={{ width: '50vw', height: '100vh', objectFit: 'contain' }} />
    );
};

export default CCTVImage;
