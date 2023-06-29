#!/bin/bash

# Run the Python scripts and save the output to respective files
python preprocessing.py > preprocessing_output.txt
python training_pipeline.py > training_pipeline_output.txt
python training_results.py > training_results_output.txt
python trials_pipeline.py > trials_pipeline_output.txt
python Statistical_U_test.py > statistical_u_test_output.txt
exit