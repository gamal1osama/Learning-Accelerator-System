#!/bin/bash
# Runs once on first container boot (empty data volume).
# POSTGRES_DB already created notes_helper; add the UI database.
set -e

psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d postgres <<-EOSQL
    CREATE DATABASE notes_helper_ui;
EOSQL
