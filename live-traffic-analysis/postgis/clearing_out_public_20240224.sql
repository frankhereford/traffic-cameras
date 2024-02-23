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
-- Name: public; Type: SCHEMA; Schema: -; Owner: pg_database_owner
--

CREATE SCHEMA public;


ALTER SCHEMA public OWNER TO pg_database_owner;

--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: pg_database_owner
--

COMMENT ON SCHEMA public IS 'standard public schema';


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
    location public.geometry(Point,4326)
);


ALTER TABLE public.detections OWNER TO postgres;

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
    public.st_chaikinsmoothing(public.st_simplify(public.st_transform(public.st_setsrid(public.st_makeline(public.st_force2d((detections.location)::public.geometry) ORDER BY detections."timestamp"), 4326), 2229), (5)::double precision), 3) AS path,
    public.st_length(public.st_chaikinsmoothing(public.st_simplify(public.st_transform(public.st_setsrid(public.st_makeline(public.st_force2d((detections.location)::public.geometry) ORDER BY detections."timestamp"), 4326), 2229), (5)::double precision), 3)) AS distance,
    EXTRACT(epoch FROM (max(detections."timestamp") - min(detections."timestamp"))) AS duration_seconds,
        CASE
            WHEN (EXTRACT(epoch FROM (max(detections."timestamp") - min(detections."timestamp"))) > (0)::numeric) THEN (((0.681818)::double precision * public.st_length(public.st_chaikinsmoothing(public.st_simplify(public.st_transform(public.st_setsrid(public.st_makeline(public.st_force2d((detections.location)::public.geometry) ORDER BY detections."timestamp"), 4326), 2229), (5)::double precision), 3))) / (EXTRACT(epoch FROM (max(detections."timestamp") - min(detections."timestamp"))))::double precision)
            ELSE (0)::double precision
        END AS average_speed_mph,
        CASE
            WHEN (EXTRACT(epoch FROM (now() - min(detections."timestamp"))) > ((60 * 5))::numeric) THEN 0
            ELSE (100 - (((EXTRACT(epoch FROM (now() - min(detections."timestamp"))) * (100)::numeric) / ((60 * 5))::numeric))::integer)
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

