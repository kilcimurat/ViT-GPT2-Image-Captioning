"""Vision transformer wrapper built on top of CLIP ViT‑L/14."""

from transformers import CLIPImageProcessor, CLIPVisionModel
import torch.nn as nn


class ViT(nn.Module):
    """Expose the CLIP ViT-L/14 encoder for feature extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def forward(self, images, return_patch_embeddings: bool = False):
        """Run CLIP preprocessing followed by the vision encoder.

        Args:
            images: A batch of images accepted by ``CLIPImageProcessor``.
            return_patch_embeddings: If ``True`` return the full patch token
                embeddings. Otherwise return the pooled CLS representation.
        """
        device = next(self.model.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        if return_patch_embeddings:
            return outputs.last_hidden_state
        return outputs.pooler_output


# model = ViT()
# torch.save(model.cpu(), "vit_model.pt")
