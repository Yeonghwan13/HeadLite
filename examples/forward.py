"""Run HeadLite on generated tensors without data, training or downloaded weights."""
import argparse
import torch
from headlite import HeadLite, HeadLiteEnsemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", action="store_true", help="use five independently initialized networks")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(77)
    device = torch.device(args.device)
    network = HeadLiteEnsemble([HeadLite() for _ in range(5)]) if args.ensemble else HeadLite()
    network = network.to(device).eval()
    inputs = (torch.randn(2, 3, 300, device=device), torch.randn(2, 3, 300, device=device),
              torch.rand(2, 40, 300, device=device), torch.randn(2, 6, device=device))
    with torch.inference_mode():
        result = network(*inputs)
    print("Generated inputs and randomly initialized weights; no performance evaluation.")
    print("parameters:", sum(p.numel() for p in network.parameters()))
    if args.ensemble:
        print("ensemble mean shape:", tuple(result.shape))
    else:
        print("mean shape:", tuple(result[0].shape), "variance shape:", tuple(result[1].shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
