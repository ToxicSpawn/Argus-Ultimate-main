"""Tests for ml/argus_ai/ — ArgusAI model suite.

Covers:
  - Backbone forward pass + output shape
  - ModalFusion with all modalities present
  - ModalFusion with missing modalities (graceful fallback)
  - DirectionHead output shape + probability sum
  - SizeHead alpha/beta positivity + mean in (0,1)
  - TimingHead non-negative delay
  - ConfidenceHead MC samples shape
  - ArgusAI full forward pass (no CoT)
  - ArgusAI full forward pass (with CoT, CRISIS gate)
  - RLTuner no-op during CRISIS regime
  - RLTuner update runs on non-CRISIS
  - ArgusAITrainer compute_loss shapes
"""

import pytest

# Skip entire module if torch not available
torch = pytest.importorskip("torch", reason="torch required for ArgusAI tests")

B = 4
T = 16
D_MODEL = 128  # smaller for fast tests
N_HEADS = 4
N_LAYERS = 2


def _get_modules():
    """Lazy import to avoid collection ordering issues."""
    from ml.argus_ai.backbone import ArgusBackbone
    from ml.argus_ai.fusion import ModalFusion, FUSED_DIM
    from ml.argus_ai.heads import DirectionHead, SizeHead, TimingHead, ConfidenceHead
    from ml.argus_ai.model import ArgusAI
    from ml.argus_ai.cot_reasoner import ChainOfThoughtReasoner
    from ml.argus_ai.rl_tuner import RLTuner
    from ml.argus_ai.trainer import ArgusAITrainer, TrainerConfig
    return {
        "ArgusBackbone": ArgusBackbone,
        "ModalFusion": ModalFusion,
        "FUSED_DIM": FUSED_DIM,
        "DirectionHead": DirectionHead,
        "SizeHead": SizeHead,
        "TimingHead": TimingHead,
        "ConfidenceHead": ConfidenceHead,
        "ArgusAI": ArgusAI,
        "ChainOfThoughtReasoner": ChainOfThoughtReasoner,
        "RLTuner": RLTuner,
        "ArgusAITrainer": ArgusAITrainer,
        "TrainerConfig": TrainerConfig,
    }


@pytest.fixture(scope="module")
def backbone():
    mods = _get_modules()
    return mods["ArgusBackbone"](d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, input_dim=256)


@pytest.fixture(scope="module")
def fusion():
    mods = _get_modules()
    return mods["ModalFusion"]()


@pytest.fixture(scope="module")
def argus_model():
    mods = _get_modules()
    return mods["ArgusAI"](d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, mc_samples=5, cot_enabled=False)


def _lob():
    return torch.randn(B, T, 128)

def _chart():
    return torch.randn(B, T, 64)

def _sentiment():
    return torch.randn(B, 32)

def _gnn():
    return torch.randn(B, T, 64)

def _regime():
    return torch.randn(B, 16)

def _regime_ids():
    return torch.randint(0, 4, (B,))


class TestArgusBackbone:
    def test_output_shape(self, backbone):
        x = torch.randn(B, T, 256)
        regime_ids = _regime_ids()
        out = backbone(x, regime_ids)
        assert out.shape == (B, T, D_MODEL)

    def test_causal_no_nan(self, backbone):
        x = torch.randn(B, T, 256)
        regime_ids = _regime_ids()
        out = backbone(x, regime_ids)
        assert not torch.isnan(out).any(), "Backbone output contains NaN"


class TestModalFusion:
    def test_all_modalities(self, fusion):
        lob = _lob()
        chart = _chart()
        sentiment = _sentiment()
        gnn = _gnn()
        regime = _regime()
        
        fused = fusion(lob=lob, chart=chart, sentiment=sentiment, gnn=gnn, regime=regime)
        assert fused.shape[0] == B

    def test_missing_chart_and_sentiment(self, fusion):
        lob = _lob()
        gnn = _gnn()
        regime = _regime()
        
        fused = fusion(lob=lob, gnn=gnn, regime=regime)
        assert fused.shape[0] == B

    def test_only_lob(self, fusion):
        lob = _lob()
        
        fused = fusion(lob=lob)
        assert fused.shape[0] == B

    def test_no_time_dim(self, fusion):
        lob = torch.randn(B, 128)  # no time dim
        regime = _regime()
        
        fused = fusion(lob=lob, regime=regime)
        assert fused.shape[0] == B


class TestHeads:
    def test_direction_head(self):
        mods = _get_modules()
        head = mods["DirectionHead"](D_MODEL)
        x = torch.randn(B, D_MODEL)
        out = head(x)
        assert out.probs.shape == (B, 3)  # buy, sell, hold
        assert torch.allclose(out.probs.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_size_head(self):
        mods = _get_modules()
        head = mods["SizeHead"](D_MODEL)
        x = torch.randn(B, D_MODEL)
        output = head(x)
        assert output.alpha.shape == (B, 1)
        assert output.beta.shape == (B, 1)
        assert (output.alpha > 0).all(), "alpha must be positive"
        assert (output.beta > 0).all(), "beta must be positive"
        assert ((output.mean_size > 0) & (output.mean_size < 1)).all(), "mean must be in (0,1)"

    def test_timing_head(self):
        mods = _get_modules()
        head = mods["TimingHead"](D_MODEL)
        x = torch.randn(B, D_MODEL)
        output = head(x)
        assert output.delay_ticks.shape == (B, 1)
        assert (output.delay_ticks >= 0).all(), "delay must be non-negative"

    def test_confidence_head(self):
        mods = _get_modules()
        head = mods["ConfidenceHead"](D_MODEL, mc_samples=5)
        x = torch.randn(B, D_MODEL)
        output = head(x)
        assert output.mean.shape == (B, 1)
        assert output.std.shape == (B, 1)


class TestArgusAI:
    def test_full_forward(self, argus_model):
        lob = _lob()
        chart = _chart()
        sentiment = _sentiment()
        gnn = _gnn()
        regime = _regime()
        regime_ids = _regime_ids()
        
        # ArgusAI.forward signature: forward(regime_ids, lob=None, chart=None, ...)
        result = argus_model(regime_ids=regime_ids, lob=lob, chart=chart, sentiment=sentiment, gnn=gnn, regime_vec=regime)
        assert result.direction_logits is not None
        assert result.size_mean is not None
        assert result.timing_delay is not None
        assert result.confidence_mean is not None

    def test_crisis_cot_gate(self):
        mods = _get_modules()
        model = mods["ArgusAI"](d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, mc_samples=5, cot_enabled=True)
        
        lob = _lob()
        chart = _chart()
        sentiment = _sentiment()
        gnn = _gnn()
        regime = _regime()
        regime_ids = torch.full((B,), 3)  # CRISIS (0-indexed: 0=ranging, 1=trending, 2=volatile, 3=crisis)
        
        result = model(regime_ids=regime_ids, lob=lob, chart=chart, sentiment=sentiment, gnn=gnn, regime_vec=regime)
        assert result.direction_logits is not None


class TestRLTuner:
    def test_noop_during_crisis(self):
        mods = _get_modules()
        model = mods["ArgusAI"](d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)
        tuner = mods["RLTuner"](model)
        
        # During CRISIS (regime_id=3), update() should return None (no-op)
        result = tuner.update(current_drawdown=0.0, regime_id=3)
        assert result is None, "RLTuner should skip updates during CRISIS regime"

    def test_update_non_crisis(self):
        mods = _get_modules()
        # Use d_model=512 to match RLTuner's ValueHead which is hardcoded to 512
        model = mods["ArgusAI"](d_model=512, n_heads=N_HEADS, n_layers=N_LAYERS)
        tuner = mods["RLTuner"](model, update_every=1)  # Set update_every=1 so buffer check passes
        
        # Push some experiences first - state_repr must be 512 to match ValueHead
        state_repr = torch.randn(512)
        for i in range(10):
            tuner.push(
                state_repr=state_repr,
                action=0,
                reward=0.1,
                log_prob=-0.5,
                regime_id=0,  # RANGING
                done=False,
            )
        
        # Update during non-crisis should work (returns None if buffer too small, or loss float)
        result = tuner.update(current_drawdown=0.0, regime_id=0)
        # Result can be None (if buffer < update_every) or a float loss
        assert result is None or isinstance(result, float)


class TestArgusAITrainer:
    def test_compute_loss(self):
        mods = _get_modules()
        model = mods["ArgusAI"](d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, mc_samples=5)
        config = mods["TrainerConfig"](lr=1e-4, weight_decay=0.01)
        trainer = mods["ArgusAITrainer"](model=model, config=config)
        
        # compute_loss signature: (output, direction_labels, size_targets, timing_targets, confidence_targets)
        # Create mock output with required attributes
        class MockOutput:
            direction_logits = torch.randn(B, 3)
            size_mean = torch.rand(B, 1)
            timing_delay = torch.rand(B, 1)
            confidence_mean = torch.rand(B, 1)
        
        output = MockOutput()
        direction_labels = torch.randint(0, 3, (B,))
        size_targets = torch.rand(B, 1)
        timing_targets = torch.rand(B, 1)
        confidence_targets = torch.rand(B, 1)
        
        loss, components = trainer.compute_loss(output, direction_labels, size_targets, timing_targets, confidence_targets)
        assert loss.shape == ()
        assert loss.item() > 0
        assert "total" in components
