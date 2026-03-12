import torch
import os
import json
import torchvision.models as models

# ---------- MANUAL ARGUMENTS FOR SPYDER ----------
class Args:
    model = "outputs/best_model.pth"            # your trained model
    output = "outputs/model_ts.pt"              # torchscript output
args = Args()
# -------------------------------------------------


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)

    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(num_ftrs, len(ckpt['classes']))
    )

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    return model, ckpt['classes']


def quantize_and_export(model, sample_input, out_path):
    # Post-training quantization for CPU
    model_cpu = model.cpu()
    qmodel = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    scripted = torch.jit.trace(qmodel, sample_input)
    scripted.save(out_path)

    print("✔ Saved quantized TorchScript model to", out_path)


def main():
    device = "cpu"

    print("Loading model from:", args.model)
    model, classes = load_checkpoint(args.model, device)

    sample = torch.randn(1, 3, 224, 224)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    quantize_and_export(model, sample, args.output)

    # save class names next to TorchScript output
    json.dump(
        classes,
        open(os.path.dirname(args.output) + "/classes.json", "w")
    )

    print("✔ Saved classes.json")


if __name__ == "__main__":
    main()
