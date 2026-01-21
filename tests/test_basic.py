import torch
from src.utils import get_device

def test_device_selection():
    device = get_device()
    assert isinstance(device, torch.device)

def test_inference_pipeline_import():
    from src.pipelines.inference_pipeline import load_prediction_model
    assert load_prediction_model is not None
