--
-- PostgreSQL database dump
--

-- Dumped from database version 15.4
-- Dumped by pg_dump version 15.6 (Ubuntu 15.6-1.pgdg22.04+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: postgis; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


--
-- Name: EXTENSION postgis; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION postgis IS 'PostGIS geometry and geography spatial types and functions';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: classes; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.classes (
    id integer NOT NULL,
    session_id integer NOT NULL,
    class_id integer NOT NULL,
    class_name text NOT NULL,
    start_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.classes OWNER TO postgres;

--
-- Name: classes_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.classes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.classes_id_seq OWNER TO postgres;

--
-- Name: classes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.classes_id_seq OWNED BY public.classes.id;


--
-- Name: detections; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.detections (
    id integer NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    session_id integer NOT NULL,
    tracker_id integer NOT NULL,
    class_id integer NOT NULL,
    image_x double precision NOT NULL,
    image_y double precision NOT NULL,
    location public.geometry(Point,2253)
);


ALTER TABLE public.detections OWNER TO postgres;

--
-- Name: detections_extended; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.detections_extended AS
 SELECT detections.id,
    detections."timestamp",
    detections.session_id,
    detections.tracker_id,
    classes.class_name,
    detections.image_x,
    detections.image_y,
    concat(detections.image_x, ', ', detections.image_y) AS image_coordinate,
    detections.location
   FROM (public.detections
     JOIN public.classes ON ((detections.class_id = classes.id)))
  ORDER BY detections.session_id DESC, detections.tracker_id DESC, detections."timestamp" DESC;


ALTER TABLE public.detections_extended OWNER TO postgres;

--
-- Name: detections_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.detections_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.detections_id_seq OWNER TO postgres;

--
-- Name: detections_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.detections_id_seq OWNED BY public.detections.id;


--
-- Name: predictions; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.predictions AS
 WITH linked_detections AS (
         WITH params AS (
                 SELECT 30 AS lag_value
                )
         SELECT detections.id,
            detections."timestamp",
            detections.session_id,
            detections.tracker_id,
            detections.class_name,
            detections.image_x,
            detections.image_y,
            detections.image_coordinate,
            detections.location,
            lag(detections."timestamp", ( SELECT params.lag_value
                   FROM params)) OVER w AS timestamp_5_detections_ago,
            lag(detections.location, ( SELECT params.lag_value
                   FROM params)) OVER w AS location_5_detections_ago,
                CASE
                    WHEN (lag(detections.location, ( SELECT params.lag_value
                       FROM params)) OVER w IS NOT NULL) THEN public.st_distance(detections.location, lag(detections.location, ( SELECT params.lag_value
                       FROM params)) OVER w)
                    ELSE NULL::double precision
                END AS distance_in_feet,
                CASE
                    WHEN (lag(detections."timestamp", ( SELECT params.lag_value
                       FROM params)) OVER w IS NOT NULL) THEN EXTRACT(epoch FROM (detections."timestamp" - lag(detections."timestamp", ( SELECT params.lag_value
                       FROM params)) OVER w))
                    ELSE NULL::numeric
                END AS time_difference_seconds,
                CASE
                    WHEN ((lag(detections.location, ( SELECT params.lag_value
                       FROM params)) OVER w IS NOT NULL) AND (lag(detections."timestamp", ( SELECT params.lag_value
                       FROM params)) OVER w IS NOT NULL)) THEN (public.st_distance(detections.location, lag(detections.location, ( SELECT params.lag_value
                       FROM params)) OVER w) / (EXTRACT(epoch FROM (detections."timestamp" - lag(detections."timestamp", ( SELECT params.lag_value
                       FROM params)) OVER w)))::double precision)
                    ELSE NULL::double precision
                END AS speed_fps,
                CASE
                    WHEN (lag(detections.location, ( SELECT params.lag_value
                       FROM params)) OVER w IS NOT NULL) THEN degrees(public.st_azimuth(lag(detections.location, ( SELECT params.lag_value
                       FROM params)) OVER w, detections.location))
                    ELSE NULL::double precision
                END AS angle_of_direction
           FROM public.detections_extended detections
          WINDOW w AS (PARTITION BY detections.session_id, detections.tracker_id ORDER BY detections."timestamp")
        )
 SELECT linked_detections.id,
    linked_detections."timestamp",
    linked_detections.session_id,
    linked_detections.tracker_id,
    linked_detections.class_name,
    linked_detections.image_x,
    linked_detections.image_y,
    linked_detections.image_coordinate,
    linked_detections.location,
    linked_detections.timestamp_5_detections_ago,
    linked_detections.location_5_detections_ago,
    linked_detections.distance_in_feet,
    linked_detections.time_difference_seconds,
    linked_detections.speed_fps,
    linked_detections.angle_of_direction,
    public.st_makeline(linked_detections.location, linked_detections.location_5_detections_ago) AS linestring_past_to_present,
    (public.st_transform((public.st_project((public.st_transform(linked_detections.location, 4326))::public.geography, ((linked_detections.speed_fps * (1)::double precision) * (((1)::numeric / 3.28084))::double precision), radians(linked_detections.angle_of_direction)))::public.geometry, 2253))::public.geometry(Point,2253) AS future_location,
    public.st_makeline(linked_detections.location, (public.st_transform((public.st_project((public.st_transform(linked_detections.location, 4326))::public.geography, ((linked_detections.speed_fps * (1)::double precision) * (((1)::numeric / 3.28084))::double precision), radians(linked_detections.angle_of_direction)))::public.geometry, 2253))::public.geometry(Point,2253)) AS linestring_present_to_future,
    true AS "true"
   FROM linked_detections
  WHERE (linked_detections.location_5_detections_ago IS NOT NULL);


ALTER TABLE public.predictions OWNER TO postgres;

--
-- Name: sessions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sessions (
    id integer NOT NULL,
    uuid text,
    start_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.sessions OWNER TO postgres;

--
-- Name: sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.sessions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.sessions_id_seq OWNER TO postgres;

--
-- Name: sessions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.sessions_id_seq OWNED BY public.sessions.id;


--
-- Name: tracked_paths; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.tracked_paths AS
 SELECT min(detections.id) AS id,
    min(detections."timestamp") AS start_time,
    max(detections."timestamp") AS end_time,
    detections.session_id,
    detections.tracker_id,
    detections.class_id,
    classes.class_name,
    public.st_chaikinsmoothing(public.st_simplify(public.st_setsrid(public.st_makeline(public.st_force2d((detections.location)::public.geometry) ORDER BY detections."timestamp"), 2253), (5)::double precision), 3) AS path,
    public.st_length(public.st_chaikinsmoothing(public.st_simplify(public.st_setsrid(public.st_makeline(public.st_force2d((detections.location)::public.geometry) ORDER BY detections."timestamp"), 2253), (5)::double precision), 3)) AS distance,
    EXTRACT(epoch FROM (max(detections."timestamp") - min(detections."timestamp"))) AS duration_seconds,
        CASE
            WHEN (EXTRACT(epoch FROM (max(detections."timestamp") - min(detections."timestamp"))) > (0)::numeric) THEN (((0.681818)::double precision * public.st_length(public.st_chaikinsmoothing(public.st_simplify(public.st_setsrid(public.st_makeline(public.st_force2d((detections.location)::public.geometry) ORDER BY detections."timestamp"), 2253), (5)::double precision), 3))) / (EXTRACT(epoch FROM (max(detections."timestamp") - min(detections."timestamp"))))::double precision)
            ELSE (0)::double precision
        END AS average_speed_mph,
        CASE
            WHEN (EXTRACT(epoch FROM (now() - min(detections."timestamp"))) > ((60 * 15))::numeric) THEN 0
            ELSE (100 - (((EXTRACT(epoch FROM (now() - min(detections."timestamp"))) * (100)::numeric) / ((60 * 15))::numeric))::integer)
        END AS minute_transparency
   FROM (public.detections
     JOIN public.classes ON ((detections.class_id = classes.id)))
  GROUP BY detections.session_id, detections.tracker_id, detections.class_id, classes.class_name
 HAVING ((count(detections.id) >= 5) AND (public.st_makeline(public.st_force2d((detections.location)::public.geometry) ORDER BY detections."timestamp") IS NOT NULL))
  ORDER BY detections.session_id, detections.tracker_id;


ALTER TABLE public.tracked_paths OWNER TO postgres;

--
-- Name: classes id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.classes ALTER COLUMN id SET DEFAULT nextval('public.classes_id_seq'::regclass);


--
-- Name: detections id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.detections ALTER COLUMN id SET DEFAULT nextval('public.detections_id_seq'::regclass);


--
-- Name: sessions id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sessions ALTER COLUMN id SET DEFAULT nextval('public.sessions_id_seq'::regclass);


--
-- Name: classes classes_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.classes
    ADD CONSTRAINT classes_pkey PRIMARY KEY (id);


--
-- Name: detections detections_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.detections
    ADD CONSTRAINT detections_pkey PRIMARY KEY (id);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: idx_classes_id_class_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_classes_id_class_name ON public.classes USING btree (id, class_name);


--
-- Name: idx_classes_session_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_classes_session_id ON public.classes USING btree (session_id);


--
-- Name: idx_detections_class_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_detections_class_id ON public.detections USING btree (class_id);


--
-- Name: idx_detections_location; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_detections_location ON public.detections USING gist (location);


--
-- Name: idx_detections_session_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_detections_session_id ON public.detections USING btree (session_id);


--
-- Name: idx_detections_session_id_tracker_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_detections_session_id_tracker_id ON public.detections USING btree (session_id, tracker_id);


--
-- Name: idx_detections_session_id_tracker_id_class_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_detections_session_id_tracker_id_class_id ON public.detections USING btree (session_id, tracker_id, class_id);


--
-- Name: idx_detections_timestamp; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_detections_timestamp ON public.detections USING btree ("timestamp");


--
-- Name: classes classes_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.classes
    ADD CONSTRAINT classes_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.sessions(id) ON DELETE CASCADE;


--
-- Name: detections detections_class_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.detections
    ADD CONSTRAINT detections_class_id_fkey FOREIGN KEY (class_id) REFERENCES public.classes(id) ON DELETE CASCADE;


--
-- Name: detections detections_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.detections
    ADD CONSTRAINT detections_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.sessions(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

