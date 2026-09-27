# C3 M4 Lab 4 — Model Quantization

This notebook walks through three PyTorch quantization techniques applied to a CNN trained on CIFAR-10, then applies dynamic quantization to a BLIP Visual Question Answering (VQA) model.

## What's in the notebook

### Part 1 — CIFAR-10 CNN quantization
1. **Setup** — imports, device selection, and the baseline `CNN` architecture (4 Conv+BatchNorm blocks, max-pooling, dropout, 3 FC layers).
2. **Load data & baseline model** — load CIFAR-10 via `helper_utils.load_cifar10()`, then load a pre-trained FP32 checkpoint and measure its size/inference time as the reference point.
3. **Dynamic quantization** — `torch.quantization.quantize_dynamic` converts `Linear`/`Conv2d` weights to `int8` with no retraining or calibration needed; fastest to apply, smallest accuracy risk mitigation.
4. **Static quantization** — a second architecture (`QuantizedCNN`) with `QuantStub`/`DeQuantStub` is prepared, calibrated on real data (observers record activation ranges), then converted to a fully `int8` model.
5. **Quantization-Aware Training (QAT)** — a third architecture (`QATCNN`) fuses Conv-BN(-ReLU) layers, fine-tunes for 5 epochs with fake-quantization inserted, then converts to the final `int8` model. This is the most involved but typically the most accurate of the three.
6. **Comparison** — size (MB) and inference time (ms) are measured after each stage and combined into one final table comparing Baseline vs. Dynamic vs. Static vs. QAT.

### Part 2 — BLIP VQA quantization
1. Load a pre-trained BLIP VQA model + processor and measure its baseline (FP32) size.
2. Upload/select an image and a question.
3. Run VQA inference with the full-precision model (answer + inference time).
4. Apply dynamic quantization to the BLIP model's `Linear` layers.
5. Re-run the same question through the quantized model.
6. Compare baseline vs. quantized: size, inference time, and generated answers side by side.

## Requirements
- Python with `torch`, `torch.quantization` (PyTorch ≥ 1.x with quantization support)
- `tqdm`
- `IPython`
- A `helper_utils` module (provided by the course) supplying:
  - `load_cifar10`, `training_loop`
  - `get_model_size`, `measure_average_inference_time_ms`, `comparison_table`, `display_full_comparison`
  - `train_qat`, `evaluate_qat`
  - `get_blip_vqa_model_and_processor`, `upload_jpg_widget`, `perform_vqa`, `blip_comparison_table`
- A pre-trained baseline checkpoint at `./baseline_pretrained_model/cifar10_cnn_30_epochs_best.pt`
- CPU or CUDA GPU (device is auto-detected; the BLIP section runs on CPU)

## Outputs produced when run
- `cifar10_cnn_quantized_dynamic.pth` — dynamically quantized CNN weights
- `cifar10_cnn_quantized_static.pth` — statically quantized CNN weights
- `cifar10_cnn_qat_quantized.pt` — QAT-quantized CNN weights

## How to use
1. Run cells top to bottom.
2. Each code cell now has a short **"What I did"** markdown note directly above it explaining its purpose — use those as a running log/commentary of the lab.
3. Swap in your own image (via the upload widget in Part 2) or edit the `question` string to try different VQA prompts.

## Key takeaway
Dynamic, static, and QAT quantization trade off implementation effort against accuracy and speed/size gains — this lab lets you measure that trade-off directly on both a small CNN and a larger vision-language model (BLIP).
