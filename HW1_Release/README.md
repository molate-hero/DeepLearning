# Homework 1

Read `hw1.pdf` for the assignment requirements. Run all commands from the
release directory.

## Setup

Use Python 3.10+ and install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Complete the TODOs in `NumPy/model.py` and `MoE/model.py` before running the
checks or training.

## Run

```sh
# NumPy gradient check and training
python NumPy/run.py --check --train

# Check the MoE implementation before training
python MoE/diagnostics.py --check

# Train and evaluate the three configurations
python MoE/train.py --mode all

# Analyze routing using the saved MoE checkpoint
python MoE/diagnostics.py
```

MoE training and routing analysis select CUDA automatically when available. Add `--device cpu`
or `--device cuda` to choose a device explicitly. To train one configuration,
replace `all` with `dense_compute`, `moe`, or `dense_parameter`.

Results are saved in `results/`: `numpy.json`, `numpy_loss.pdf`,
`language.json`, and `moe_seed416.pt`. Training accepts `--out` for a custom
output directory; pass the same path to diagnostics with `--results`.
Rerunning training replaces the corresponding results.
