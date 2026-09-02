# AI-Powered Pharmacovigilance Case Intake Assistant

A simplified, interview-focused implementation of an AI-assisted Pharmacovigilance (PV) case intake pipeline.

## Overview

This project demonstrates how Large Language Models (LLMs) can assist Pharmacovigilance case processors by extracting structured safety information from unstructured case documents and supporting downstream MedDRA coding and narrative generation.

The implementation is intentionally simplified for interview and educational purposes. The full production-oriented implementation is maintained separately and is demonstrated through the deployed application.

## Problem Statement

Pharmacovigilance case intake often involves reviewing unstructured source documents and identifying key safety information such as:

- Adverse events
- Medical history
- Medications
- Laboratory and diagnostic information

This information then needs to be structured and prepared for downstream Pharmacovigilance processing, including medical coding and case narrative preparation.

## Solution

The pipeline combines Python, OpenAI GPT models, and the BioPortal MedDRA terminology service to demonstrate an AI-assisted intake workflow.

### Pipeline

PDF document  
↓  
Text extraction  
↓  
Text cleaning  
↓  
LLM-based structured extraction  
↓  
MedDRA candidate search  
↓  
AI-assisted LLT selection  
↓  
PT lookup  
↓  
Structured PV case  
↓  
PV narrative generation

## Key Features

- PDF text extraction using PyMuPDF
- Text cleaning and preprocessing
- Structured PV information extraction using OpenAI
- Source-supported adverse event extraction
- BioPortal MedDRA candidate retrieval
- AI-assisted MedDRA LLT selection
- LLT → PT lookup
- Structured case output
- PV narrative generation

## Technology Stack

- Python
- OpenAI API
- PyMuPDF
- BioPortal / MedDRA
- Requests
- python-dotenv

## Project Structure

```text
AI-PV-Case-Intake-Assistant-Public/
│
├── pv.py
├── README.md
├── requirements.txt
└── .gitignore

## Scope

This repository contains a simplified implementation of the project for technical interview discussion and demonstration.

The complete application includes additional processing, validation, business rules, and deployment components that are not included in this public version.
