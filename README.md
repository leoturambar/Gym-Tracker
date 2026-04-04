# Gym Tracker

App personale per il tracciamento degli allenamenti con analisi AI.

## Features
- Log sessioni con pre-popolamento automatico
- Calcolo Relative Training Volume (RTV) per gruppo muscolare
- Radar muscolare con confronto temporale
- Analisi narrativa con Claude AI

## Stack
Python · Streamlit · Pandas · Plotly · Anthropic API

## Setup
```bash
conda create -n gymtracker python=3.11
conda activate gymtracker
pip install streamlit pandas matplotlib plotly anthropic
streamlit run app.py
```