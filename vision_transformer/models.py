import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def create_model(num_classes, pretrained=True):
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    try:
        model = vit_b_16(weights=weights)
    except Exception as e:  # most common cause: no internet for the first weight download
        raise RuntimeError("Could not load the pretrained ViT-B/16 weights (needs internet the first "
                           f"time, ~330 MB). Original error: {e}")
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Linear(in_features, 512), nn.GELU(), nn.Dropout(0.4),
        nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def set_head_only(model, head_only):
    for p in model.parameters():
        p.requires_grad = not head_only
    for p in model.heads.parameters():
        p.requires_grad = True


def _layer_id(name, depth):
    if name.startswith(("class_token", "conv_proj", "encoder.pos_embedding")):
        return 0
    if name.startswith("encoder.layers.encoder_layer_"):
        return int(name.split("encoder_layer_")[1].split(".")[0]) + 1
    return depth + 1  # final layer-norm + head


def layerwise_param_groups(model, lr, layer_decay, weight_decay):
    """AdamW param groups with layer-wise LR decay (lower blocks change more slowly) and
    no weight decay on biases / norms / embeddings."""
    depth = len(model.encoder.layers)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = _layer_id(name, depth)
        no_decay = p.ndim == 1 or name in ("class_token", "encoder.pos_embedding")
        key = (lid, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * layer_decay ** (depth + 1 - lid),
                           "weight_decay": 0.0 if no_decay else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())
