import numpy as np
import torch
from .. import util


class DDPMSampler():
    def __init__(self, n_inference_steps=50, n_training_steps=1000, beta_start=0.00085, beta_end=0.0120):
        self.n_inference_steps = n_inference_steps
        self.n_training_steps = n_training_steps
        
        # Create beta schedule
        self.betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=np.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        # Calculate variance for sampling
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # Create timestep schedule for inference
        # Use evenly spaced timesteps from the training schedule
        step_ratio = self.n_training_steps // self.n_inference_steps
        self.timesteps = np.arange(0, self.n_training_steps, step_ratio)[:self.n_inference_steps]
        self.timesteps = self.timesteps[::-1].copy()  # Reverse for denoising direction
        
        self.step_count = 0

    def get_input_scale(self, step_count=None):
        """Get the scaling factor for input latents at the current timestep"""
        if step_count is None:
            step_count = self.step_count
        
        if step_count >= len(self.timesteps):
            return 1.0
        
        timestep = self.timesteps[step_count]
        alpha_cumprod = self.alphas_cumprod[timestep]
        return 1.0 / np.sqrt(alpha_cumprod)

    def set_strength(self, strength=1):
        """Set the strength of denoising by starting from a later timestep"""
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.step_count = start_step

    def step(self, latents, output, add_noise=True):
        """
        Perform one denoising step using DDPM sampling
        
        Args:
            latents: Current noisy latents
            output: Predicted noise from the U-Net
            add_noise: Whether to add random noise (set to False for final step)
        """
        t = self.step_count
        self.step_count += 1
        
        if t >= len(self.timesteps):
            return latents
        
        timestep = self.timesteps[t]
        
        # Get current and previous alpha values
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        alpha_t = self.alphas[timestep]
        beta_t = self.betas[timestep]
        
        if t > 0:
            prev_timestep = self.timesteps[t - 1] if t < len(self.timesteps) else 0
            alpha_cumprod_t_prev = self.alphas_cumprod[prev_timestep]
        else:
            alpha_cumprod_t_prev = 1.0
        
        # Calculate predicted original sample (x_0)
        pred_original_sample = (latents - np.sqrt(1 - alpha_cumprod_t) * output) / np.sqrt(alpha_cumprod_t)
        
        # Calculate coefficients for x_t-1
        pred_sample_coeff = np.sqrt(alpha_cumprod_t_prev) * beta_t / (1 - alpha_cumprod_t)
        current_sample_coeff = np.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)
        
        # Calculate predicted previous sample mean
        pred_prev_sample = pred_sample_coeff * pred_original_sample + current_sample_coeff * latents
        
        # Add noise if not the final step
        if add_noise and t < len(self.timesteps) - 1:
            # Calculate variance
            variance = self.posterior_variance[timestep]
            if variance > 0:
                # Generate random noise with the same shape as latents
                if isinstance(latents, torch.Tensor):
                    noise = torch.randn_like(latents)
                else:
                    noise = np.random.randn(*latents.shape).astype(latents.dtype)
                pred_prev_sample += np.sqrt(variance) * noise
        
        return pred_prev_sample

    def add_noise(self, original_samples, noise, timestep):
        """
        Add noise to original samples at a given timestep (forward diffusion process)
        
        Args:
            original_samples: Clean samples
            noise: Gaussian noise to add
            timestep: Timestep for noise level
        """
        alpha_cumprod = self.alphas_cumprod[timestep]
        sqrt_alpha_cumprod = np.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - alpha_cumprod)
        
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        return noisy_samples

    def get_timesteps(self):
        """Get the current timestep schedule"""
        return self.timesteps.copy()

    def reset(self):
        """Reset the sampler to initial state"""
        self.step_count = 0