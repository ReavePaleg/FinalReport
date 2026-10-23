"""
Denoise synthetic EDMR spectra and compare methods.

CHATGPT was used to help write the neural network section of code
CHATGPT was used to fix errors/rewrite pieces of code
"""

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import torch
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

x = np.arange(1024)

# %% Adding noise

noise_rng = np.random.default_rng(42)

NOISE_BETA = {
    "brown": 2,
    "pink": 1,
    "white": 0,
    "blue": -1,
    "violet": -2,
}

def gaussian_noise(sigma, rng=None):
    if rng is None:
        rng = noise_rng
    return rng.normal(0, sigma, size=x.shape)


def coloured_noise(sigma, colour="pink", rng=None):
    """Generate noise with power proportional to f**(-beta)."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return np.zeros(x.shape, dtype=float)

    if colour == "white":
        return gaussian_noise(sigma, rng)

    beta = NOISE_BETA[colour]
    white = gaussian_noise(1, rng)
    frequencies = np.fft.rfftfreq(len(x))

    # Fourier weighting: https://stackoverflow.com/questions/67085963
    # Amplitude is the square root of the power spectrum.
    weights = np.zeros_like(frequencies)
    weights[1:] = frequencies[1:] ** (-beta / 2)

    result = np.fft.irfft(
        np.fft.rfft(white) * weights,
        n=len(x),
    )

    result -= result.mean()
    std = result.std()
    if std == 0:
        return np.zeros_like(result)
    return sigma * result / std

def mse(true, pred):
    return np.mean((true - pred) ** 2)

# %% Creating the dataset

def normalise_spectra(spectra):
    """Normalise one spectrum or a batch using the noisy amplitude."""
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim not in (1, 2) or spectra.size == 0:
        raise ValueError("Expected a spectrum [length] or batch [samples, length].")
    if not np.isfinite(spectra).all():
        raise ValueError("Spectra must contain only finite values.")

    # The peak term stops narrow, noise-free peaks giving a near-zero scale.
    absolute = np.abs(spectra)
    percentile_scale = np.percentile(absolute, 95, axis=-1, keepdims=True)
    peak_scale = np.max(absolute, axis=-1, keepdims=True).astype(np.float64) / 10
    scales = np.maximum(percentile_scale, peak_scale).astype(np.float32)
    scales = np.where(scales > 0, scales, np.float32(1.0))
    normalised = spectra / scales
    if not np.isfinite(normalised).all():
        raise ValueError("Normalisation produced non-finite input values.")
    return normalised, scales

def make_clean_spectrum(rng, max_resonances=5, max_width=40, max_height=1.5):
    """Sum derivative-Gaussian resonances with random widths and heights."""
    clean = np.zeros(len(x))

    for _ in range(rng.integers(1, max_resonances + 1)):
        width = rng.uniform(0, max_width)
        centre = rng.uniform(x[0] - max_width, x[-1] + max_width)
        height = rng.uniform(0, max_height)

        if width == 0:
            continue

        # Extrema are at centre +/- width, with amplitudes +/- height.
        z = (x - centre) / width
        clean += height * z * np.exp(0.5 - 0.5 * z**2)

    return clean

def make_noise_spectrum(rng, colours):
    # Allow noise-free spectra as well as mixtures.
    n_components = rng.integers(0, len(colours) + 1)
    selected_colours = rng.choice(colours, size=n_components, replace=False)
    noise = np.zeros(len(x))

    for colour in selected_colours:
        strength = rng.uniform(0, 1.0)
        noise += coloured_noise(strength, colour, rng)

    # Set the total noise level after mixing the colours.
    sigma = rng.uniform(0, 0.4)
    noise -= noise.mean()
    std = noise.std()
    if sigma == 0 or std == 0:
        return np.zeros_like(noise)
    return sigma * noise / std

def make_drift_spectrum(
    rng,
    offset_max=0.3,
    slope_max=0.3,
    curvature_max=0.2,
    wave_max=0.15,
):
    # Offset, slope, curvature and a slow sinusoid, in synthetic signal units.
    u = np.linspace(0, 1, len(x))
    t = 2 * u - 1
    offset = rng.uniform(-offset_max, offset_max)
    slope = rng.uniform(-slope_max, slope_max)
    curvature = rng.uniform(-curvature_max, curvature_max)
    wave_amplitude = rng.uniform(0, wave_max)
    cycles = rng.uniform(0, 1.5)
    phase = rng.uniform(0, 2 * np.pi)

    drift = offset + slope * t
    drift += curvature * (t**2 - np.mean(t**2))
    drift += wave_amplitude * np.sin(2 * np.pi * cycles * u + phase)
    return drift

def make_dataset(
    n_samples,
    seed,
    colours=("white", "pink", "brown", "blue", "violet"),
    drift_probability=0.8,
):
    if not 0 <= drift_probability <= 1:
        raise ValueError("drift_probability must be between 0 and 1")
    rng = np.random.default_rng(seed)

    clean = np.empty((n_samples, len(x)), dtype=np.float32)
    noisy = np.empty_like(clean)

    for i in range(n_samples):
        clean[i] = make_clean_spectrum(rng)

        noise = make_noise_spectrum(rng, colours)
        drift = np.zeros(len(x))
        if rng.random() < drift_probability:
            drift = make_drift_spectrum(rng)

        noisy[i] = clean[i] + noise + drift

    # Normalise both using the noisy spectrum.
    noisy_normalised, scales = normalise_spectra(noisy)
    clean_normalised = clean / scales
    if not np.isfinite(clean_normalised).all():
        raise ValueError("Normalisation produced non-finite clean targets.")
    return TensorDataset(
        torch.from_numpy(noisy_normalised),
        torch.from_numpy(clean_normalised),
        torch.from_numpy(scales),
    )

# Regenerating the data invalidates any previous training.
training_complete = False

# These ranges and noise levels describe synthetic data, not measured scans.
train_dataset = make_dataset(50000, seed=10)
val_dataset = make_dataset(2000, seed=20)
test_dataset = make_dataset(2000, seed=30)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
)

val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False
)

X_batch, y_batch, scale_batch = next(iter(train_loader))
print(X_batch.shape, y_batch.shape, X_batch.dtype)

# %% One clean/noisy example

example_input, example_target, example_scale = train_dataset[0]
plt.figure(figsize=(10, 3))
plt.plot(x, (example_input * example_scale).numpy(), color="grey", alpha=0.7, label="Noisy input")
plt.plot(x, (example_target * example_scale).numpy(), color="black", label="Clean target")
plt.xlabel("Sample index")
plt.ylabel("Signal (original units)")
plt.title("One clean spectrum and its noisy input")
plt.legend()
plt.tight_layout()
plt.show()

# %% The neural network

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available() else "cpu"
)
print(f"Using {device} device")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels):
        super().__init__()
        self.projection = nn.Conv1d(in_channels, skip_channels, kernel_size=1)
        self.conv = ConvBlock(2 * skip_channels, skip_channels)

    def forward(self, x, skip):
        x = F.interpolate(
            x, size=skip.shape[-1], mode="linear", align_corners=False
        )
        x = self.projection(x)
        return self.conv(torch.cat((x, skip), dim=1))

# U-Net: https://arxiv.org/abs/1505.04597
# Residual denoising: https://arxiv.org/abs/1608.03981
class NeuralNetwork(nn.Module):
    """Predict noise and drift, then subtract them from the input."""
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(1, 16)
        self.encoder2 = ConvBlock(16, 32)
        self.encoder3 = ConvBlock(32, 64)
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.bottleneck = ConvBlock(64, 128)

        self.decoder3 = UpBlock(128, 64)
        self.decoder2 = UpBlock(64, 32)
        self.decoder1 = UpBlock(32, 16)
        self.noise_head = nn.Conv1d(16, 1, kernel_size=1)

        # Zero residual makes the initial prediction equal to the input.
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)

    def forward(self, noisy):
        # Conv1d expects [batch, channels, length].
        x = noisy.unsqueeze(1)
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        x = self.bottleneck(self.pool(skip3))

        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        predicted_noise = self.noise_head(x).squeeze(1)
        return noisy - predicted_noise

torch.manual_seed(42)
model = NeuralNetwork().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% Training

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_mse = 0.0
    total_samples = 0

    for X, y, scale in dataloader:
        X, y, scale = X.to(device), y.to(device), scale.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite training loss.")

        # Optimise normalised MSE, but report errors in original units.
        original_mse = ((pred.detach() - y) * scale).square().mean()
        loss.backward()
        optimizer.step()

        total_mse += original_mse.item() * X.size(0)
        total_samples += X.size(0)

    return total_mse / total_samples

def evaluate(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for X, y, scale in dataloader:
            X, y, scale = X.to(device), y.to(device), scale.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite validation loss.")

            total_loss += loss.item() * X.size(0)
            total_mse += ((pred - y) * scale).square().mean().item() * X.size(0)
            total_samples += X.size(0)

    # Use normalised loss for checkpoint selection.
    return total_loss / total_samples, total_mse / total_samples

training_complete = False

epochs = 40
patience = 6
min_delta = 1e-5
train_mses = []
val_mses = []

best_val_loss = float("inf")
best_val_mse = None
best_weights = None
best_epoch = None
early_stop_best = float("inf")
epochs_without_improvement = 0

for epoch in range(epochs):
    train_mse = train(train_loader, model, loss_fn, optimizer)
    val_loss, val_mse = evaluate(val_loader, model, loss_fn)

    if not np.isfinite([train_mse, val_loss, val_mse]).all():
        raise ValueError("Training or validation loss is not finite.")

    train_mses.append(train_mse)
    val_mses.append(val_mse)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_mse = val_mse
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1

    # Patience uses min_delta; the best checkpoint above does not.
    if val_loss < early_stop_best - min_delta:
        early_stop_best = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Training MSE: {train_mse:.6f} | "
        f"Validation MSE: {val_mse:.6f}"
    )

    if epochs_without_improvement >= patience:
        print(f"Early stopping: no meaningful improvement for {patience} epochs.")
        break

model.load_state_dict(best_weights)
model.eval()
print(f"Restored epoch {best_epoch} | Validation MSE: {best_val_mse:.6f}")
training_complete = True

# %% Denoising a spectrum

def denoise_spectra(spectra, model, batch_size=128):
    """Denoise one raw spectrum or a batch, returning original units."""
    normalised, scales = normalise_spectra(spectra)
    single_spectrum = normalised.ndim == 1
    if single_spectrum:
        normalised = normalised[None, :]

    model_device = next(model.parameters()).device
    was_training = model.training
    predictions = []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(normalised), batch_size):
                inputs = torch.from_numpy(
                    normalised[start:start + batch_size]
                ).to(model_device)
                predictions.append(model(inputs).cpu().numpy())
    finally:
        model.train(was_training)

    denoised = np.concatenate(predictions, axis=0) * scales
    return denoised[0] if single_spectrum else denoised

plt.figure(figsize=(7, 4))
epoch_numbers = range(1, len(train_mses) + 1)
plt.plot(epoch_numbers, train_mses, label="Training")
plt.plot(epoch_numbers, val_mses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE (original amplitude units squared)")
plt.legend()
plt.tight_layout()
plt.show()

# %% Comparing denoising methods

def comparison_arrays(dataset):
    """Read inputs, targets and scales from a fixed data split."""
    inputs, targets, scales = (
        tensor.detach().cpu().numpy().astype(np.float64)
        for tensor in dataset.tensors
    )
    return inputs, targets, scales

def remove_polynomial_baseline(spectra, degree):
    """Subtract a polynomial fitted to each noisy spectrum."""
    coordinate = np.linspace(-1, 1, spectra.shape[-1])
    basis = np.polynomial.polynomial.polyvander(coordinate, degree)
    coefficients = spectra @ np.linalg.pinv(basis).T
    return spectra - coefficients @ basis.T

def traditional_denoise(spectra, method, parameters):
    spectra = np.asarray(spectra, dtype=np.float64)
    if method == "Moving average":
        return uniform_filter1d(
            spectra, size=parameters["window"], axis=-1, mode="reflect"
        )
    if method == "Gaussian smoothing":
        return gaussian_filter1d(
            spectra, sigma=parameters["sigma"], axis=-1, mode="reflect"
        )
    if method in ("Polynomial baseline", "Baseline + Savitzky–Golay"):
        spectra = remove_polynomial_baseline(spectra, parameters["degree"])
    if method == "Polynomial baseline":
        return spectra
    if method in ("Savitzky–Golay", "Baseline + Savitzky–Golay"):
        return ss.savgol_filter(
            spectra, window_length=parameters["window"],
            polyorder=parameters["polyorder"], axis=-1, mode="interp"
        )
    raise ValueError(f"Unknown traditional method: {method}")

# Tune these on validation data, then keep them fixed for the test set.
comparison_windows = [5, 11, 21, 41, 81, 121]
comparison_candidates = {
    "Moving average": [{"window": w} for w in comparison_windows],
    "Gaussian smoothing": [{"sigma": s} for s in [1, 2, 4, 8, 16, 24]],
    "Savitzky–Golay": [
        {"window": w, "polyorder": order}
        for w in comparison_windows for order in [2, 3]
    ],
    "Polynomial baseline": [{"degree": d} for d in [0, 1, 2, 3]],
    "Baseline + Savitzky–Golay": [
        {"degree": d, "window": w, "polyorder": order}
        for d in [0, 1, 2, 3]
        for w in comparison_windows for order in [2, 3]
    ],
}

# %% Choose settings using validation data

comparison_X_val, comparison_y_val, val_scales = comparison_arrays(val_dataset)
selected_parameters = {}
validation_mses = {}
for method, candidates in comparison_candidates.items():
    losses = np.array([
        mse(comparison_y_val, traditional_denoise(comparison_X_val, method, settings))
        for settings in candidates
    ])
    if not np.isfinite(losses).all():
        raise ValueError(f"Non-finite validation error from {method}.")
    selected_parameters[method] = dict(candidates[int(np.argmin(losses))])
    prediction = traditional_denoise(comparison_X_val, method, selected_parameters[method])
    validation_mses[method] = mse(comparison_y_val * val_scales, prediction * val_scales)

best_traditional_method = min(validation_mses, key=validation_mses.get)

print(pd.DataFrame({
    "Selected parameters": {
        method: str(parameters) for method, parameters in selected_parameters.items()
    }
}).rename_axis("Method").to_string())

# %% Evaluate on the test set

comparison_X_test, comparison_y_test, test_scales = comparison_arrays(test_dataset)
comparison_X_test = comparison_X_test * test_scales
comparison_y_test = comparison_y_test * test_scales
comparison_predictions = {"Noisy input": comparison_X_test}

for method, parameters in selected_parameters.items():
    comparison_predictions[method] = traditional_denoise(
        comparison_X_test, method, parameters
    )

if globals().get("training_complete", False):
    model.load_state_dict(best_weights)
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("The best network weights are not finite.")
    comparison_predictions["Neural network"] = denoise_spectra(comparison_X_test, model)
else:
    print("Network comparison pending: complete training and rerun this cell.")

comparison_rows = []
for method, predictions in comparison_predictions.items():
    if predictions.shape != comparison_y_test.shape or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite and match the test targets.")
    comparison_rows.append({"Method": method, "MSE": mse(comparison_y_test, predictions)})
comparison_results = pd.DataFrame(comparison_rows).set_index("Method")
print(f"Unnormalised MSE: {len(comparison_X_test):,} held-out test spectra")
print(comparison_results.to_string(float_format="{:.6f}".format))

# %% Compare errors and reconstructed spectra

plt.figure(figsize=(8, 4))
plt.barh(comparison_results.index, comparison_results["MSE"], color="steelblue")
plt.gca().invert_yaxis()
plt.xlim(left=0)
plt.xlabel("MSE (original amplitude units squared)")
plt.title("Denoising error on 2,000 held-out test spectra")
plt.tight_layout()
plt.show()


comparison_example_indices = [0, 1, 2]
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, index in zip(axes, comparison_example_indices):
    ax.plot(x, comparison_X_test[index], color="grey", alpha=0.4, label="Noisy input")
    ax.plot(x, comparison_y_test[index], color="black", linewidth=1.8, label="Clean target")
    ax.plot(x, comparison_predictions[best_traditional_method][index],
            color="tab:blue", label=f"Best traditional: {best_traditional_method}")
    if "Neural network" in comparison_predictions:
        ax.plot(x, comparison_predictions["Neural network"][index],
                color="tab:red", label="Neural network")
    ax.set_title(f"Test spectrum {index}")
    ax.set_ylabel("Signal (original units)")
axes[0].legend()
axes[-1].set_xlabel("Sample index")
plt.tight_layout()
plt.show()
