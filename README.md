# Web Reader App

A web application that extracts the main content from web pages, removes most advertisements and irrelevant elements, and presents clean, readable text.

## Current Features

- Extract article title
- Extract main article content
- Remove most advertisements and navigation elements
- Estimate reading time
- Simple Gradio web interface
- Provider-neutral text-to-speech with Kokoro as the default provider
- Automatic Chinese/English voice selection and adjustable speech speed

## Text-to-speech setup

Use Python 3.11 (3.12 is also accepted by the dependency marker) and install
`requirements.txt`. Kokoro and its current ML dependency stack did not work in the
Python 3.13 Colab runtime, so the optional Kokoro dependency is deliberately skipped
there and the app reports an actionable compatibility message. This keeps the core
article extractor importable on 3.13 instead of failing at startup. Piper remains a
registered fallback provider in the provider-neutral service, but Kokoro is the app
default.
