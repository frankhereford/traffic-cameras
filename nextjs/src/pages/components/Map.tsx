import React, { useState, useCallback } from 'react';
import { GoogleMap, useJsApiLoader, LatLngBoundsLiteral } from '@react-google-maps/api';

interface MapProps {
    center: LatLngBoundsLiteral;
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
        const bounds = new window.google.maps.LatLngBounds(center);
        map.fitBounds(bounds);
        setMap(map);
    }, [center]);

    const onUnmount = useCallback(() => {
        setMap(null);
    }, []);

    console.log("isLoaded", isLoaded)

    return isLoaded ? (
        <GoogleMap
            mapContainerStyle={containerStyle}
            center={center}
            zoom={10}
            onLoad={onLoad}
            onUnmount={onUnmount}
        >
            { /* Child components, such as markers, info windows, etc. */}
            <></>
        </GoogleMap>
    ) : <></>
}

export default Map;