# Optimizing Reliability in Grounded Generation: A Comprehensive Framework for Detecting and Mitigating Factual Inconsistencies in Large Language Models

## Overview
This repository contains a comprehensive fact-checking framework designed to infer and mitigate factual inconsistencies using the Llama 70B instruct model deployed on IBM BAM. The project is structured into modules for detection, mitigation, and end-to-end processing of datasets, specifically tailored for the FeTaQA dataset.

## Repository Structure
The repository is organized into the following folders and files:

### Detection_Module
This module is responsible for detecting factual inconsistencies in the provided dataset.

1. **detection.py**: Script for inferring the Llama 70B instruct model for the detection task.
2. **Complete_dataset_detection_input.csv**: Preprocessed FeTaQA dataset in a compatible format for the detection prompt.
3. **Complete_dataset_detection_output.csv**: Output of the detection prompt for all entries in the input CSV file.

### Mitigation_Module
This module focuses on mitigating the detected inconsistencies in the dataset.

1. **mitigation.py**: Script for inferring the Llama 70B instruct model for the mitigation task.
2. **Complete_dataset_mitigation_input.csv**: Preprocessed FeTaQA dataset in a compatible format for the mitigation prompt.
3. **Complete_dataset_mitigation_output.csv**: Output of the mitigation prompt for all entries in the input CSV file.

### End_to_End_Framework
This module integrates both detection and mitigation tasks into an end-to-end framework.

1. **detection&mitigation.py**: Script for inferring the Llama 70B instruct model for the entire end-to-end framework.
2. **end_to_end_testing_data.csv**: Preprocessed FeTaQA dataset in a compatible format for the end-to-end framework.
3. **End_to_end_results.txt**: Output of the first 100 entries of the input CSV file.
4. **detection&mitigation.ipynb**: Jupyter notebook containing the same code as detection&mitigation.py, for easy trial and demo of the framework.

### Dataset_curation
This folder contains scripts to convert the original FeTaQA dataset into a format compatible with the detection and mitigation prompts.

1. **synthetic_FI_induction.py**: Script for synthetically inducing inconsistencies in the sentences of the FeTaQA dataset responses.
2. **Other files**: Additional scripts and tools for dataset curation.

## Additional Files
- **.env**: Environment file where you can set your BAM API key by replacing `<GENAI_KEY>`.
- **Full_data_all_modules_final.csv**: Contains all the outputs at all checkpoints of the framework for all entries of the formatted FeTaQA dataset.

## Usage
1. **Setup**: Ensure you have your BAM API key set in the `.env` file.
2. **Detection**: Run the detection module using `detection.py` with the provided input CSV.
3. **Mitigation**: Run the mitigation module using `mitigation.py` with the provided input CSV.
4. **End-to-End**: Execute the end-to-end framework using `detection&mitigation.py` or the corresponding Jupyter notebook.

## Contact
For any questions or issues, please contact [Pradhuman.Tiwari@ibm.com].
