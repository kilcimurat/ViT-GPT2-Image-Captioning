"""Vision transformer wrapper built on top of CLIP ViT‑L/14."""

from transformers import CLIPImageProcessor, CLIPVisionModel
import torch.nn as nn


class ViT(nn.Module):
    """Expose the CLIP ViT-L/14 encoder for feature extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def forward(self, images):
        """Run CLIP preprocessing followed by the vision encoder."""
        device = next(self.model.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state


# model = ViT()
# torch.save(model.cpu(), "vit_model.pt")
