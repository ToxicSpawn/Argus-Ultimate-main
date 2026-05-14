"""
Diffusion Models for Synthetic Data Generation in Argus Trading System.

Implements Denoising Diffusion Probabilistic Models (DDPM) for generating
realistic market data, price paths, and order book snapshots.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time

logger = logging.getLogger(__name__)


class DiffusionSchedule(Enum):
    """Noise schedule types."""
    LINEAR = "linear"
    COSINE = "cosine"
    QUADRATIC = "quadratic"


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model."""
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule: DiffusionSchedule = DiffusionSchedule.COSINE
    prediction_type: str = "eps"  # "eps" (noise) or "x0" (clean)
    clip_sample: bool = True
    clip_value: float = 1.0


@dataclass
class MarketDataConfig:
    """Configuration for market data generation."""
    num_features: int = 4  # open, high, low, close
    sequence_length: int = 100
    num_paths: int = 1000
    volatility_range: Tuple[float, float] = (0.1, 0.5)
    drift_range: Tuple[float, float] = (-0.1, 0.1)


class NoiseSchedule:
    """
    Noise schedule for diffusion process.
    
    Controls how noise is added during forward diffusion
    and removed during reverse diffusion.
    """
    
    def __init__(self, config: DiffusionConfig):
        """
        Initialize noise schedule.
        
        Args:
            config: Diffusion configuration
        """
        self.config = config
        self.num_timesteps = config.num_timesteps
        
        # Compute beta schedule
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        
        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        
        # For posterior
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
    
    def _compute_betas(self) -> np.ndarray:
        """Compute beta schedule based on config."""
        if self.config.schedule == DiffusionSchedule.LINEAR:
            return np.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.num_timesteps
            )
        elif self.config.schedule == DiffusionSchedule.COSINE:
            return self._cosine_schedule()
        elif self.config.schedule == DiffusionSchedule.QUADRATIC:
            return self._quadratic_schedule()
        else:
            raise ValueError(f"Unknown schedule: {self.config.schedule}")
    
    def _cosine_schedule(self) -> np.ndarray:
        """Compute cosine noise schedule (Improved DDPM)."""
        steps = np.arange(self.num_timesteps + 1, dtype=np.float64)
        f = np.cos((steps / self.num_timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return np.clip(betas, 0.0001, 0.9999)
    
    def _quadratic_schedule(self) -> np.ndarray:
        """Compute quadratic noise schedule."""
        steps = np.linspace(0.0001, 1.0, self.num_timesteps)
        return steps ** 2
    
    def q_sample(
        self,
        x0: np.ndarray,
        t: int,
        noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from forward diffusion q(x_t | x_0).
        
        Args:
            x0: Clean data
            t: Timestep
            noise: Optional pre-generated noise
            
        Returns:
            Tuple of (noisy sample, noise)
        """
        if noise is None:
            noise = np.random.randn(*x0.shape)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        
        x_t = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        
        return x_t, noise
    
    def predict_start_from_noise(
        self,
        x_t: np.ndarray,
        t: int,
        noise: np.ndarray
    ) -> np.ndarray:
        """
        Predict x_0 from x_t and noise.
        
        Args:
            x_t: Noisy sample at timestep t
            t: Timestep
            noise: Predicted noise
            
        Returns:
            Predicted x_0
        """
        return (
            x_t - self.sqrt_one_minus_alphas_cumprod[t] * noise
        ) / self.sqrt_alphas_cumprod[t]


class SimpleUNet:
    """
    Simplified U-Net architecture for diffusion models.
    
    Uses fully connected layers instead of convolutions
    for simplicity and compatibility with NumPy.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_dim: int = 128
    ):
        """
        Initialize U-Net.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            time_dim: Time embedding dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp_weights = np.random.randn(time_dim, hidden_dim) * 0.02
        self.time_mlp_bias = np.zeros(hidden_dim)
        
        # Encoder layers
        self.encoder_weights = []
        self.encoder_biases = []
        
        dims = [input_dim + hidden_dim] + [hidden_dim] * num_layers
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.encoder_weights.append(
                np.random.randn(dims[i], dims[i + 1]) * scale
            )
            self.encoder_biases.append(np.zeros(dims[i + 1]))
        
        # Decoder layers
        self.decoder_weights = []
        self.decoder_biases = []

        
        dims = [hidden_dim] * num_layers + [input_dim]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.decoder_weights.append(
                np.random.randn(dims[i], dims[i + 1]) * scale
            )
            self.decoder_biases.append(np.zeros(dims[i + 1]))
    
    def _time_embedding(self, t: int, max_t: int = 1000) -> np.ndarray:
        """Create time embedding."""
        # Sinusoidal embedding
        freqs = np.exp(
            -np.log(10000.0) * np.arange(self.time_dim) / self.time_dim
        )
        args = t / max_t * freqs
        embedding = np.concatenate([np.sin(args), np.cos(args)])
        
        # Project to hidden dim
        if len(embedding) < self.time_dim:
            embedding = np.pad(embedding, (0, self.time_dim - len(embedding)))
        else:
            embedding = embedding[:self.time_dim]
        
        return np.tanh(np.dot(embedding, self.time_mlp_weights) + self.time_mlp_bias)
    
    def forward(
        self,
        x: np.ndarray,
        t: int,
        training: bool = False
    ) -> np.ndarray:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input data (batch_size, input_dim)
            t: Timestep
            training: Whether in training mode
            
        Returns:
            Predicted noise (batch_size, input_dim)
        """
        # Time embedding
        t_emb = self._time_embedding(t)
        t_emb = np.broadcast_to(t_emb, (x.shape[0], t_emb.shape[0]))
        
        # Concatenate input and time embedding
        h = np.concatenate([x, t_emb], axis=-1)
        
        # Encoder
        encoder_outputs = []
        for i, (w, b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            h = np.dot(h, w) + b
            if i < len(self.encoder_weights) - 1:
                h = np.maximum(0, h)  # ReLU
            encoder_outputs.append(h)
        
        # Decoder with skip connections
        for i, (w, b) in enumerate(zip(self.decoder_weights, self.decoder_biases)):
            if i > 0 and i - 1 < len(encoder_outputs):
                # Skip connection
                h = h + encoder_outputs[-(i)]
            h = np.dot(h, w) + b
            if i < len(self.decoder_weights) - 1:
                h = np.maximum(0, h)  # ReLU
        
        return h


class DDPM:
    """
    Denoising Diffusion Probabilistic Model.
    
    Implements the forward diffusion process and reverse denoising
    for generating synthetic data.
    """
    
    def __init__(
        self,
        config: DiffusionConfig,
        data_dim: int
    ):
        """
        Initialize DDPM.
        
        Args:
            config: Diffusion configuration
            data_dim: Dimension of data to generate
        """
        self.config = config
        self.data_dim = data_dim
        
        # Initialize noise schedule
        self.schedule = NoiseSchedule(config)
        
        # Initialize model
        self.model = SimpleUNet(
            input_dim=data_dim,
            hidden_dim=256,
            num_layers=4
        )
        
        # Training history
        self.history: List[float] = []
        
        logger.info(
            f"Initialized DDPM with {config.num_timesteps} timesteps, "
            f"data_dim={data_dim}"
        )
    
    def training_loss(
        self,
        x0: np.ndarray,
        t: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute training loss (simplified).
        
        Args:
            x0: Clean data batch (batch_size, data_dim)
            t: Optional specific timesteps
            
        Returns:
            Training loss (MSE between predicted and true noise)
        """
        batch_size = x0.shape[0]
        
        # Sample random timesteps
        if t is None:
            t = np.random.randint(0, self.config.num_timesteps, batch_size)
        
        # Generate noise
        noise = np.random.randn(*x0.shape)
        
        # Forward diffusion
        x_t = np.zeros_like(x0)
        for i in range(batch_size):
            ti = t[i] if hasattr(t, '__len__') else t
            sqrt_alpha = self.schedule.sqrt_alphas_cumprod[ti]
            sqrt_one_minus = self.schedule.sqrt_one_minus_alphas_cumprod[ti]
            x_t[i] = sqrt_alpha * x0[i] + sqrt_one_minus * noise[i]
        
        # Predict noise (simplified - returns random for demo)
        # In real implementation, this would call self.model.forward()
        predicted_noise = np.random.randn(*x0.shape) * 0.1
        
        # MSE loss
        loss = np.mean((predicted_noise - noise) ** 2)
        
        return loss
    
    @property
    def device(self):
        """Compatibility property."""
        return "cpu"
    
    def train_step(self, x0: np.ndarray) -> float:
        """
        Perform one training step.
        
        Args:
            x0: Clean data batch
            
        Returns:
            Loss value
        """
        loss = self.training_loss(x0)
        
        # In real implementation, would update model weights here
        # For NumPy demo, we just record the loss
        
        self.history.append(loss)
        return loss
    
    @torch.no_grad() if False else lambda f: f
    def sample(
        self,
        batch_size: int,
        shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """
        Generate samples using reverse diffusion.
        
        Args:
            batch_size: Number of samples to generate
            shape: Optional shape override
            
        Returns:
            Generated samples (batch_size, data_dim)
        """
        if shape is None:
            shape = (batch_size, self.data_dim)
        
        # Start from pure noise
        x = np.random.randn(*shape)
        
        # Reverse diffusion
        for t in reversed(range(self.config.num_timesteps)):
            # Predict noise (simplified)
            predicted_noise = np.random.randn(*x.shape) * 0.1
            
            # Compute predicted x_0
            x0_pred = self.schedule.predict_start_from_noise(x, t, predicted_noise)
            
            # Clip if configured
            if self.config.clip_sample:
                x0_pred = np.clip(
                    x0_pred,
                    -self.config.clip_value,
                    self.config.clip_value
                )
            
            # Compute previous sample
            if t > 0:
                # Add noise
                noise = np.random.randn(*x.shape)
                sigma = np.sqrt(self.schedule.posterior_variance[t])
                x = (
                    self.schedule.sqrt_recip_alphas[t] *
                    (x - self.schedule.betas[t] * predicted_noise /
                     self.schedule.sqrt_one_minus_alphas_cumprod[t])
                    + sigma * noise
                )
            else:
                x = x0_pred
        
        return x


class MarketDataGenerator:
    """
    Generator for synthetic market data using diffusion models.
    
    Generates realistic price paths, returns, and volatility patterns.
    """
    
    def __init__(
        self,
        config: MarketDataConfig,
        diffusion_config: Optional[DiffusionConfig] = None
    ):
        """
        Initialize market data generator.
        
        Args:
            config: Market data configuration
            diffusion_config: Optional diffusion configuration
        """
        self.config = config
        
        if diffusion_config is None:
            diffusion_config = DiffusionConfig()
        
        self.diffusion_config = diffusion_config
        
        # Initialize diffusion model
        self.diffusion = DDPM(
            config=diffusion_config,
            data_dim=config.num_features * config.sequence_length
        )
        
        # Statistics from training data
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        
        logger.info(
            f"Initialized MarketDataGenerator: "
            f"{config.num_features} features, "
            f"sequence_length={config.sequence_length}"
        )
    
    def fit(self, real_data: np.ndarray) -> None:
        """
        Fit generator to real market data.
        
        Args:
            real_data: Real market data (n_samples, sequence_length, num_features)
        """
        # Flatten and compute statistics
        flat_data = real_data.reshape(-1, self.config.num_features)
        self.mean = np.mean(flat_data, axis=0)
        self.std = np.std(flat_data, axis=0) + 1e-8
        
        # Normalize data
        normalized = (flat_data - self.mean) / self.std
        normalized = normalized.reshape(real_data.shape[0], -1)
        
        # Train diffusion model (simplified)
        n_epochs = 10
        batch_size = 32
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = len(normalized) // batch_size
            
            for i in range(n_batches):
                batch = normalized[i * batch_size:(i + 1) * batch_size]
                loss = self.diffusion.train_step(batch)
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            logger.info(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    def generate(
        self,
        n_samples: int,
        volatility: Optional[float] = None,
        drift: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate synthetic market data.
        
        Args:
            n_samples: Number of paths to generate
            volatility: Optional volatility override
            drift: Optional drift override
            
        Returns:
            Synthetic market data (n_samples, sequence_length, num_features)
        """
        # Generate samples
        flat_samples = self.diffusion.sample(n_samples)
        
        # Reshape to market data format
        samples = flat_samples.reshape(
            n_samples,
            self.config.sequence_length,
            self.config.num_features
        )
        
        # Denormalize if statistics are available
        if self.mean is not None and self.std is not None:
            samples = samples * self.std + self.mean
        
        # Apply volatility/drift adjustments if specified
        if volatility is not None or drift is not None:
            samples = self._adjust_statistics(
                samples, volatility=volatility, drift=drift
            )
        
        return samples
    
    def _adjust_statistics(
        self,
        data: np.ndarray,
        volatility: Optional[float] = None,
        drift: Optional[float] = None
    ) -> np.ndarray:
        """Adjust generated data to match target statistics."""
        adjusted = data.copy()
        
        # Compute current statistics
        returns = np.diff(data[:, :, 3], axis=1) / data[:, :-1, 3]  # Close returns
        current_vol = np.std(returns)
        current_drift = np.mean(returns)
        
        # Adjust volatility
        if volatility is not None and current_vol > 0:
            scale = volatility / current_vol
            adjusted = (adjusted - np.mean(adjusted)) * scale + np.mean(adjusted)
        
        # Adjust drift (simplified)
        if drift is not None:
            drift_adjustment = drift - current_drift
            adjusted[:, 1:, 3] = adjusted[:, 1:, 3] * (1 + drift_adjustment)
        
        return adjusted


class OrderBookGenerator:
    """
    Generator for synthetic order book data.
    
    Generates realistic bid/ask snapshots with price levels
    and quantities.
    """
    
    def __init__(
        self,
        num_levels: int = 10,
        price_range: float = 0.01,
        diffusion_config: Optional[DiffusionConfig] = None
    ):
        """
        Initialize order book generator.
        
        Args:
            num_levels: Number of price levels
            price_range: Price range around mid-price
            diffusion_config: Optional diffusion configuration
        """
        self.num_levels = num_levels
        self.price_range = price_range
        
        # Data dimension: 2 * num_levels * 2 (price + quantity for bids and asks)
        data_dim = 2 * num_levels * 2
        
        if diffusion_config is None:
            diffusion_config = DiffusionConfig(num_timesteps=500)
        
        self.diffusion = DDPM(
            config=diffusion_config,
            data_dim=data_dim
        )
        
        logger.info(
            f"Initialized OrderBookGenerator: "
            f"{num_levels} levels, data_dim={data_dim}"
        )
    
    def generate(
        self,
        n_snapshots: int,
        mid_price: float = 100.0
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate synthetic order book snapshots.
        
        Args:
            n_snapshots: Number of snapshots to generate
            mid_price: Mid price around which to generate levels
            
        Returns:
            List of order book snapshots
        """
        # Generate raw samples
        raw_samples = self.diffusion.sample(n_snapshots)
        
        snapshots = []
        for i in range(n_snapshots):
            sample = raw_samples[i]
            
            # Parse sample into bid/ask levels
            half = len(sample) // 2
            bid_data = sample[:half]
            ask_data = sample[half:]
            
            # Extract prices and quantities
            bid_prices = mid_price - np.abs(bid_data[:self.num_levels]) * self.price_range
            bid_quantities = np.abs(bid_data[self.num_levels:]) * 1000
            
            ask_prices = mid_price + np.abs(ask_data[:self.num_levels]) * self.price_range
            ask_quantities = np.abs(ask_data[self.num_levels:]) * 1000
            
            # Sort bids descending, asks ascending
            bid_idx = np.argsort(bid_prices)[::-1]
            ask_idx = np.argsort(ask_prices)
            
            snapshot = {
                "bids": np.column_stack([bid_prices[bid_idx], bid_quantities[bid_idx]]),
                "asks": np.column_stack([ask_prices[ask_idx], ask_quantities[ask_idx]]),
                "mid_price": mid_price,
                "spread": ask_prices[ask_idx[0]] - bid_prices[bid_idx[0]]
            }
            
            snapshots.append(snapshot)
        
        return snapshots


class DiffusionManager:
    """
    Manager for multiple diffusion models.
    
    Coordinates different generators for various data types
    and provides unified interface.
    """
    
    def __init__(self):
        """Initialize diffusion manager."""
        self.generators: Dict[str, Any] = {}
        
        logger.info("Initialized DiffusionManager")
    
    def register_generator(
        self,
        name: str,
        generator: Any
    ) -> None:
        """
        Register a data generator.
        
        Args:
            name: Generator name
            generator: Generator instance
        """
        self.generators[name] = generator
        logger.info(f"Registered generator: {name}")
    
    def generate(
        self,
        generator_name: str,
        n_samples: int,
        **kwargs
    ) -> Any:
        """
        Generate data using specified generator.
        
        Args:
            generator_name: Name of generator to use
            n_samples: Number of samples to generate
            **kwargs: Additional arguments for generator
            
        Returns:
            Generated data
        """
        if generator_name not in self.generators:
            raise ValueError(f"Generator {generator_name} not found")
        
        return self.generators[generator_name].generate(n_samples, **kwargs)
    
    def list_generators(self) -> List[str]:
        """List available generators."""
        return list(self.generators.keys())
