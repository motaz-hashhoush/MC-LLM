--
-- PostgreSQL database dump
--

-- Dumped from database version 17.5
-- Dumped by pg_dump version 17.5

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

DROP DATABASE IF EXISTS llm_logs;
--
-- Name: llm_logs; Type: DATABASE; Schema: -; Owner: -
--

CREATE DATABASE llm_logs WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'English_United States.1252';


\connect llm_logs

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: request_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.request_logs (
    id uuid NOT NULL,
    task_type character varying(32) NOT NULL,
    input_text text NOT NULL,
    output_text text,
    status character varying(16) NOT NULL,
    tokens_used integer,
    latency_ms double precision,
    error_message text,
    created_at timestamp with time zone NOT NULL,
    completed_at timestamp with time zone
);


--
-- Data for Name: request_logs; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.request_logs (id, task_type, input_text, output_text, status, tokens_used, latency_ms, error_message, created_at, completed_at) FROM stdin;
\.


--
-- Name: request_logs request_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.request_logs
    ADD CONSTRAINT request_logs_pkey PRIMARY KEY (id);


--
-- Name: ix_request_logs_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_request_logs_status ON public.request_logs USING btree (status);


--
-- Name: ix_request_logs_task_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_request_logs_task_type ON public.request_logs USING btree (task_type);


--
-- PostgreSQL database dump complete
--

