# A Knowledge Injection Evaluation Framework for Improving Factual Accuracy in Generative Language Models
*Master’s Thesis - Adrian Oberföll*

## 1. Setup
- NVIDIA GPU is necessary
- Install *requirements_uc2.txt* when running on UC2 with *jupyter/base/2023-10-10 module*
    - Python 3.9.7
    - CUDA 12.0
    - Torch 2.0.1
- Full list of requirements: *requirements_full.txt*
- Download the weights for MEND from https://rome.baulab.info/data/weights/
    - GPT-2: *mend-10tok-gpt2-xl.pt*
    - GPT-J: *mend-10tok-gpt-j-6b.pt*
    - Move files to *experiments/data/mend/weights*
- Data for ROME is automatically retrieved when executing the algorithm for the first time

## 2. Prepare datasets
- Run *datasets/retrieve_wikidata.ipynb* to retrieve necessary entity data from Wikidata
- Run *datasets/fewrel/preprocess_fewrel.ipynb* to create FewRel evaluation dataset
- Run *datasets/counterfact/preprocess_counterfact.ipynb* to create CounterFact evaluation dataset

## 3. Knowledge Injection
- ``cd experiments``
- Run *expriments.py* with necessary parameters, for example:
``python experiments.py --method=in-context --model_name=gpt2-xl --dataset=fewrel --split=val --output_dir="results/ici"``
- model_name corresponds to filename of hparams JSON in *methods/hparams/[method_name]*
- JSON lines file with generation before and after injection for each sample is saved in *output_dir*

## 4. Evaluation Framework
- ``cd experiments``
- Run *eval_framework.py* with necessary parameters to compute injection accuracy metrics, *Fluency*, *Grammar*, and *Time*
- For example: ``python eval_framework.py --output_dir=results/ici``
- *eval_framework.py* reads all JSON files in the created output folder when running *experiments.py*
- VRAM consumption is output in the log when running the script
- Results are written to a .csv file in *experiments/results*

## 5. Pipeline for Factual Error Correction
- ``cd experiments``
- Run *pipeline.py* with necessary parameters, for example: ``python pipeline.py --method=rome --model_name=gpt-j-6b --dataset=fewrel --split=val --output_dir=results/pipeline``
- additional params:
    - ``--persist_wikidata``: save retrieved Wikidata triples for next runs
    - ``--injection_model=[model_name]``: user different model for injection when using ICI
    - ``--retrieve_gt``: inject ground truth dataset triples instead of retrieved triples
- JSON lines file with results is saved to *output_dir*

### Pipeline Evaluation
- *experiments/eval_pipeline.ipynb*

## Other
### Injection Accuracy Metric Experiment
- *experiments/injection_accuracy_eval.ipynb*

### ICI few-shot prompt
- *experiments/few_shot_prompt.ipynb*

### Train MEND hyper-networks
- ``cd methods/mend``
- ``python -m run +alg=mend +experiment=gen +model=gpt2large``
- Trained model weights are saved to *methods/mend/output*
- Add .pt file extension to model file
