# COMP8240 - Easy Data Augmentation Reproduction and Extension

This repository contains the code, constructed data and result files used for the COMP8240 final project, **Reproducing Easy Data Augmentation for Text Classification on New Data**.

## Project scope

The project is based on Wei and Zou (2019), *EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks*.

The repository contains:

- a targeted SST-2 CNN/BiLSTM reproduction with and without EDA-style augmentation;
- a constructed 105-example customer-service intent dataset;
- fixed train/validation/test splits;
- a TF-IDF + logistic-regression baseline/EDA experiment;
- augmentation-sensitivity results;
- the 28-pair LLM-as-judge audit reported in the final report.

The targeted neural run is **not** claimed to be an exact reproduction of the full original paper. It uses 100 SST-2 training examples and trainable 300-dimensional embeddings rather than the original GloVe 840B embeddings. The final report explains this limitation.

## Repository structure

```text
src/
  run_sst2_targeted.py
  run_customer_service_experiment.py

data/
  customer_service_full_105.csv
  customer_service_train.csv
  customer_service_validation.csv
  customer_service_test.csv
  customer_service_eda_input.txt

results/
  final_experiment_results.csv
  augmentation_sensitivity.csv
  sst2_targeted_neural_runs.csv
  sst2_targeted_neural_summary.csv
  llm_judge_audit.csv
  llm_judge_prompt.txt
```

## Running the customer-service experiment

From the repository root:

```bash
pip install -r requirements.txt
python src/run_customer_service_experiment.py
```

The script reads the fixed train/test data and reproduces the baseline versus EDA-style comparison described in the report.

## Running the targeted SST-2 neural experiment

```bash
python src/run_sst2_targeted.py
```

This reproduces the targeted CNN/BiLSTM sanity check used in the report. The script contains the exact 100 training and 100 validation examples used for that constrained run so that the experiment is self-contained.

## Main reported results

Customer-service intent classification:

- baseline accuracy: 76.2%
- EDA-style mean accuracy across five seeds: 84.8%
- baseline macro-F1: 0.760
- EDA-style mean macro-F1: 0.835

Targeted SST-2 neural check across five seeds:

- CNN: 48.4% baseline -> 52.2% with EDA
- BiLSTM: 43.8% baseline -> 54.6% with EDA

## Data and ethics

The customer-service data is synthetic/constructed for this project. It does not contain real customer records or personally identifying information. The small size and controlled language are limitations discussed in the report.

## Reference

Wei, J. and Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1670/
