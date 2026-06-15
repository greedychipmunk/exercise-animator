# Agent Guide - Exercise Animator

This file serves as a guide for AI agents working on this project. It outlines the project structure, configuration, and developer workflows.

## Project Overview
The Exercise Animator is a Gradio application that takes the name of an exercise and generates an instructional animation using the Wan2.1 text-to-video model. The application is hosted as a Hugging Face Space.

## Tech Stack
- **Framework**: Python 3
- **UI & API**: Gradio, Gradio Client
- **Text-to-Video Model**: `fffiloni/Wan2.1` via Gradio Client API

## Repository Structure
- `app.py`: The entrypoint Gradio application containing UI layout and Hugging Face API client integration.
- `requirements.txt`: Python package dependencies.
- `README.md`: Space metadata and user-facing documentation.
- `AGENTS.md`: Technical guide for AI coding assistants (this file).
- `.github/workflows/sync-to-hub.yml`: GitHub Actions workflow for automated deployment.

## Local Development Setup

### Installation
To install the dependencies locally, run:
```bash
pip install -r requirements.txt
```

### Running the App
To run the Gradio app locally:
```bash
python app.py
```
This will launch a local server (typically at `http://127.0.0.1:7860`).

## Deployment Workflow
The project is configured for continuous deployment to Hugging Face Spaces.

### GitHub Actions
The deployment workflow is defined in `.github/workflows/sync-to-hub.yml`.

- **Trigger**: Every push to the `main` branch.
- **Action**: Uses the `huggingface/hub-sync` action to mirror the repository contents to the Hugging Face Space.

### Prerequisites for Deployment
1. A Hugging Face Space named `exercise-animator` under the user account (e.g. `greedychipmunk/exercise-animator`).
2. A Hugging Face write token named `HF_TOKEN` must be added to the GitHub repository's Action secrets (`Settings > Secrets and variables > Actions > New repository secret`).
