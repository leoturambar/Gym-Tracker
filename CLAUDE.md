# Gym Tracker — Claude Code Context

## What this is
Personal training log with AI-powered workout analysis. Tracks sessions,
computes Relative Training Volume (RTV) per muscle group, and generates
narrative coaching via LLM.

## Stack
Python · Streamlit · Pandas · Plotly · Anthropic API (claude-sonnet) · Ollama

## Project structure
- app.py — Streamlit UI and tab layout
- config.py — exercise library, muscle mapping
- data_manager.py — session and bodyweight read/write
- metrics.py — RTV calculations, muscle scoring
- llm.py — Claude API calls and prompt building
- launch.bat — Windows launcher
- data/ — personal training data (gitignored, never touch)

## Key logic
RTV (Relative Training Volume) weights sets by proximity to failure — it is
the core metric, implemented in metrics.py. Do not simplify or replace it
without explicit instruction.

Four AI coaching modes are implemented and working in llm.py:
general analysis, balance assessment, progression review, next session
suggestion. Results vary with data volume — this is expected.

## LLM backend
Uses Anthropic API (claude-sonnet) by default. Ollama is supported as
fallback. Backend selection is in config.py.

## Rules
- Write all code and docstrings in English
- Never modify or read data/ files — personal training data, gitignored
- Never modify config files that contain API keys or credentials
- Do not push sensitive data, keys, or personal preferences to git
- Ask before making structural changes to the UI layout
- When adding features, follow the existing module separation:
  UI logic in app.py, data in data_manager.py, metrics in metrics.py,
  LLM calls in llm.py