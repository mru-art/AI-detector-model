---
title: SCRAM Detector
emoji: 🎨
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# S.C.R.A.M detector API

Estimates whether an artwork is human-made or AI-generated. This Space is the backend for the S.C.R.A.M website.

- `POST /detect` with a multipart `file`: the verdict, P(AI), the probability for each source, and a region map
- `GET /health`: model status
- `GET /docs`: interactive API docs

Results are estimates, not proof.
