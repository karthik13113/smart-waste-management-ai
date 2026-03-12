# Smart Waste Management — Code Package

This package contains working scripts for:
- Training an image classifier (PyTorch) on TrashNet-like datasets.
- Evaluating the model and exporting a TorchScript quantized model for edge deployment.
- A simple edge inference script (PyTorch/TorchScript).
- A routing simulation using OR-Tools for telemetry-driven collection.

## Structure
- `dataset.py` — Dataset loader for image-folder structure.
- `train.py` — Training script (PyTorch).
- `eval.py` — Evaluation script and confusion matrix saving.
- `export_torchscript.py` — Quantize and export TorchScript model.
- `edge_infer.py` — Inference script that loads TorchScript model for edge.
- `routing_simulation.py` — Simulate bins and run CVRP with OR-Tools.
- `requirements.txt` — Python packages needed.

## How to use
1. Prepare data: dataset folder structured as:
   ```
   data/
     train/
       plastic/
       paper/
       glass/
       metal/
       cardboard/
       trash/
     val/
     test/
   ```
   Due to size limitations, the full dataset is not uploaded.  
2. Create a Python environment and install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Train:
   ```
   python train.py --data_dir data --epochs 30 --batch_size 32 --out_dir outputs
   ```
4. Evaluate:
   ```
   python eval.py --data_dir data --model outputs/best_model.pth --out outputs
   ```
5. Export TorchScript quantized:
   ```
   python export_torchscript.py --model outputs/best_model.pth --output outputs/model_ts.pt
   ```
6. Edge inference:
   ```
   python edge_infer.py --model outputs/model_ts.pt --image sample.jpg
   ```
7. Routing simulation:
   ```
   python routing_simulation.py
   ```

## Notes
- This is a baseline pipeline. You can customize augmentation, architecture, and hyperparameters.
- For production edge deployment you may want to convert to TFLite or PyTorch Mobile and perform model optimizations.
