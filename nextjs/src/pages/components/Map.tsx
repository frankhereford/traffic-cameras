import React, { useState, useCallback } from 'react';
import { GoogleMap, useJsApiLoader } from '@react-google-maps/api';

type Center = {
    lat: number;
    lng: number;
};

interface MapProps {
    coordinates: Array<{ cctv: [number, number], map: [number, number] }>;
    center: Center;
    containerStyle: {
        width: string;
        height: string;
    };
}

const Map: React.FC<MapProps> = ({ center, containerStyle }) => {
    const { isLoaded } = useJsApiLoader({
        id: 'google-map-script',
        googleMapsApiKey: "AIzaSyAcbnyfHzwzLinnwjgapc7eMOg22yXkmuY"
    });

    const [map, setMap] = useState<google.maps.Map | null>(null);

    const onLoad = useCallback((map: google.maps.Map) => {
        const circle = new window.google.maps.Circle({
            center: center,
            radius: 200 * 0.3048, // Convert feet to meters
        });

        const bounds = circle.getBounds();

        map.fitBounds(bounds!);
        setMap(map);
    }, [center]);

    const onUnmount = useCallback(() => {
        setMap(null);
    }, []);

    return isLoaded ? (
        <GoogleMap
            mapContainerStyle={containerStyle}
            center={center}
            zoom={9}
            onLoad={onLoad}
            onUnmount={onUnmount}
            options={{ tilt: 0, mapTypeId: 'satellite' }} // Set tilt to 0 for a top-down view
        >
            { /* Child components, such as markers, info windows, etc. */}
            <></>
        </GoogleMap>
    ) : <></>
}

export default Map;