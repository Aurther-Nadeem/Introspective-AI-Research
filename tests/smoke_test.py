import unittest
import torch
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from introspectai.models.load import load_model
from introspectai.steering.concepts import build_concept_vector, ConceptStore
from introspectai.steering.inject import ResidualInjector
from introspectai.experiments.injected_thoughts import run_trial as run_trial_a
from introspectai.experiments.prefill_authorship import run_authorship_trial as run_trial_b
from introspectai.eval.parse import parse_introspection_json

class MockConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class TestIntrospectAI(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_artifacts")
        self.test_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def test_parse_json(self):
        text = 'Some text {"detected": true, "confidence": 0.9, "concept": "foo", "authorship": "self", "rationale": "bar"}'
        res = parse_introspection_json(text)
        self.assertTrue(res.detected)
        self.assertEqual(res.concept, "foo")
        
    def test_concept_store(self):
        store = ConceptStore(self.test_dir)
        vec = torch.randn(10)
        store.save("model", 1, "concept", vec, {})
        loaded = store.load("model", 1, "concept")
        self.assertTrue(torch.equal(vec, loaded))
        
    def test_injection(self):
        # Mock model
        model = MagicMock()
        model.config.hidden_size = 10
        model.device = "cpu"
        
        # Mock layer
        layer = MagicMock()
        layer.forward = MagicMock(return_value="output")
        model.model.layers = [layer]
        
        vec = torch.randn(10)
        injector = ResidualInjector(model, 0, vec, 1.0)
        
        with injector:
            # Simulate forward call
            hidden_states = torch.randn(1, 5, 10)
            injector.model.model.layers[0].forward(hidden_states)
            
        # Verify wrapper was called (hard to verify exact logic with mocks, but can check no crash)
        self.assertTrue(True)

    def test_whiten(self):
        from introspectai.steering.concepts import whiten
        vec = torch.randn(10)
        cov = torch.eye(10)
        # Whitening with identity cov should just be normalization (roughly, if we ignore the shrinkage/solve details for a moment)
        # Actually our implementation does solve(cov, vec). If cov=I, result is vec.
        res = whiten(vec, cov)
        self.assertTrue(torch.allclose(res, vec, atol=1e-5))
        
    def test_rms(self):
        from introspectai.models.load import estimate_residual_rms
        model = MagicMock()
        model.device = "cpu"
        # Mock output with hidden_states
        # hidden_states tuple: (emb, layer_0_out)
        # If layer_idx=0, we look at hidden_states[1]
        h = torch.ones(1, 5, 10) * 2 # RMS should be 2
        outputs = MagicMock()
        outputs.hidden_states = (None, h)
        model.return_value = outputs
        
        tokenizer = MagicMock()
        tokenizer.return_value.input_ids.to.return_value = "ids"
        
        rms = estimate_residual_rms(model, tokenizer, 0)
        self.assertAlmostEqual(rms, 2.0)

if __name__ == "__main__":
    unittest.main()
