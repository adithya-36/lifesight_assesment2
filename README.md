# Lifesight Assessment 2: Marketing Mix Modeling

## Overview

This repository contains the implementation of a Marketing Mix Modeling (MMM) pipeline, aimed at quantifying the impact of various marketing channels on revenue. The project includes data preprocessing, model training, evaluation, and visualization components.

## Repository Structure
lifesight2\marketing-mix-model
├── data\
│   ├── raw\
│   │   ├── .gitkeep
│   │   └── mmm_weekly.csv
│   └── processed\
│       └── .gitkeep
├── src\
│   ├── data\
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── data_preprocessor.py
│   ├── models\
│   │   ├── __init__.py
│   │   ├── mmm_model.py
│   │   └── model_validation.py
│   ├── visualization\
│   │   ├── __init__.py
│   │   └── plotting.py
│   └── utils\
│       ├── __init__.py
│       └── helpers.py
├── notebooks\
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_model_evaluation.ipynb
├── tests\
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_mmm_model.py
├── config\
│   └── config.yaml
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
