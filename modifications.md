# Code Modifications and Next Steps

## Modifications

- **`experiments/experiments.py`**:
  - Added the cache directory.
  - Enabled print statements for rome and constrained beam search to better track what is happening during execution.

- **`methods/ft/ft_main.py`**:
  - Added print statements to display the data type of the weight matrices during fine-tuning.

- **`methods/rome/compute_u.py`**:
  - Adjusted the precision to match the model's parameter tensors.

- **`methods/rome/compute_v.py`**:
  - Matched the naming convention of the embedding dimension for the Mistral model.

- **`methods/rome/rome_main.py`**:
  - Added checks for the size of the left and right vectors being computed.

- **New File**: `methods/hparams/in-context/Qwen2.5-0.5B-Instruct.json`
  - Added this file to test the instruction prompt template for the Owen models.

## Next Steps

1. Setup in-context prompt templates for the Owen and Llama 3 models discussed.
2. Run the models mentioned in isolation and as part of the pipeline.
