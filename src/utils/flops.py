import torch
from fvcore.nn import FlopCountAnalysis

def get_model_training_gflops(model, adaptation_strategy: str, input_size=(1, 6, 512, 512)):
    """
    Computes both Forward and Backward pass GFLOPs analytically for the model.
    Returns:
        (fwd_gflops, bwd_gflops) as floats.
    """
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)

    # Calculate total forward FLOPs
    total_analysis = FlopCountAnalysis(model, x)
    total_analysis.unsupported_ops_warnings(False)
    fwd_total = total_analysis.total() / 1e9

    # Check if the model has a separated backbone encoder
    if hasattr(model, "encoder") and model.encoder is not None:
        encoder_analysis = FlopCountAnalysis(model.encoder, x)
        encoder_analysis.unsupported_ops_warnings(False)
        fwd_encoder = encoder_analysis.total() / 1e9
        fwd_decoder = max(0.0, fwd_total - fwd_encoder)
    else:
        # If it's a baseline trained from scratch (like UNet)
        fwd_encoder = fwd_total
        fwd_decoder = 0.0

    # Calculate backward pass GFLOPs analytically
    if adaptation_strategy == "linear_probe":
        # Linear Probing: Encoder is fully frozen. Only decoder requires backward pass.
        # Decoder: 2x forward FLOPs (1x activation gradients, 1x weight gradients).
        bwd_gflops = 2.0 * fwd_decoder
    elif adaptation_strategy == "lora":
        # LoRA: Encoder weights frozen, but activations backpropagated to reach adapters.
        # Encoder: 1x forward FLOPs (activation gradients only).
        # Decoder: 2x forward FLOPs (activation + weight gradients).
        bwd_gflops = (1.0 * fwd_encoder) + (2.0 * fwd_decoder)
    else:
        # Full Fine-Tuning / Trained from scratch: All parameters trainable.
        # Entire model: 2x forward FLOPs.
        bwd_gflops = 2.0 * fwd_total

    return round(float(fwd_total), 4), round(float(bwd_gflops), 4)
