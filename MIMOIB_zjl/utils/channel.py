import torch

def power_normalize(feature):
    in_shape = feature.shape
    batch_size = in_shape[0]
        
    # Flatten feature tensor to 2D: (batch_size, features)
    z_in = feature.reshape(batch_size, -1)
        
    # Calculate signal power: |z|^2
    sig_pwr = torch.square(torch.abs(z_in))
        
    # Calculate average signal power for each sample
    ave_sig_pwr = sig_pwr.mean(dim=1, keepdim=True)  # Keep dimension for broadcasting
        
    # Avoid division by zero
    ave_sig_pwr = torch.clamp(ave_sig_pwr, min=1e-10)
        
    # Power normalization: z_normalized = z / sqrt(average_power)
    z_in_norm = z_in / torch.sqrt(ave_sig_pwr)
        
    # Restore original shape
    inputs_in_norm = z_in_norm.reshape(in_shape)
        
    return inputs_in_norm


def MIMO_channel_rician(x, SNR, Nr, Nt, K_rician=10):
    """
    Rician MIMO channel model
        
    Args:
        x: Input signal (batch_size, 2*K) where first K dims are real part, last K dims are imaginary part
        SNR: Signal-to-noise ratio (dB)
        Nr: Number of receive antennas
        Nt: Number of transmit antennas  
        K_rician: Rician K-factor
        
    Returns:
        y: Received signal (batch_size, 2*Nr)
    """
        
    SNR_linear = 10 ** (SNR / 10)
    K_linear = torch.tensor(K_rician, dtype=torch.float32)
        
    # Split real and imaginary parts
    batch_size = x.size(0)
    K = x.size(1) // 2
    x_real = x[:, :K]  # First K dims (real part)
    x_imag = x[:, K:]  # Last K dims (imaginary part)
        
    # Create complex signal: x_complex shape (batch_size, K)
    x_complex = torch.complex(x_real, x_imag)
        
    # Calculate input signal power (should be close to 1 after power_normalize)
    signal_power = torch.mean(torch.abs(x_complex)**2, dim=1, keepdim=True)  # (batch_size, 1)
        
    # Generate Rician channel matrix H
    # LOS component (deterministic component)
    H_los_real = torch.ones(batch_size, Nr, Nt)
    H_los_imag = torch.zeros(batch_size, Nr, Nt)
    H_los = torch.complex(H_los_real, H_los_imag)
        
    # NLOS component (Rayleigh fading)
    H_nlos_real = torch.randn(batch_size, Nr, Nt) / torch.sqrt(torch.tensor(2.0))
    H_nlos_imag = torch.randn(batch_size, Nr, Nt) / torch.sqrt(torch.tensor(2.0))
    H_nlos = torch.complex(H_nlos_real, H_nlos_imag)
        
    # Combine LOS and NLOS components
    H = torch.sqrt(K_linear / (K_linear + 1)) * H_los + torch.sqrt(1 / (K_linear + 1)) * H_nlos
        
    # Channel matrix normalization (maintain unit average power gain)
    H_power = torch.sum(torch.abs(H)**2, dim=(1, 2)) / (Nr * Nt)  # (batch_size,)
    H = H / torch.sqrt(H_power.unsqueeze(-1).unsqueeze(-1))
        
    # Move to correct device
    H = H.to(x.device)
        
    # Channel transmission: y_signal = H * x
    y_signal = torch.bmm(H, x_complex.unsqueeze(-1)).squeeze(-1)  # (batch_size, Nr)
        
    # Calculate received signal power (actual signal power after channel)
    received_signal_power = torch.mean(torch.abs(y_signal)**2, dim=1, keepdim=True)  # (batch_size, 1)
        
    # Calculate noise variance based on SNR and actual received signal power
    # SNR = Signal_Power / Noise_Power  =>  Noise_Power = Signal_Power / SNR
    noise_variance = received_signal_power / SNR_linear  # (batch_size, 1)
        
    # Generate complex Gaussian noise
    # For complex noise, real and imaginary parts each have variance of noise_variance/2
    n_real = torch.randn(batch_size, Nr, device=x.device) * torch.sqrt(noise_variance / 2)
    n_imag = torch.randn(batch_size, Nr, device=x.device) * torch.sqrt(noise_variance / 2)
    n = torch.complex(n_real, n_imag)
        
    # Final received signal: y = H*x + n
    y_complex = y_signal + n
        
    # Split complex output into real and imaginary parts format (2*Nr dimensions)
    y_real = torch.real(y_complex)  # Shape: (batch_size, Nr)
    y_imag = torch.imag(y_complex)  # Shape: (batch_size, Nr)
        
    # Concatenate real and imaginary parts: (batch_size, 2*Nr)
    y = torch.cat([y_real, y_imag], dim=1)
        
    return y, H, noise_variance

def MIMO_channel_AWGN(x, SNR, Nr, Nt):
    """
    AWGN channel model
        
    Args:
        x: Input signal (batch_size, 2*K) where first K dims are real part, last K dims are imaginary part
        SNR: Signal-to-noise ratio (dB)
        Nr: Number of receive antennas
        Nt: Number of transmit antennas
        
    Returns:
        y: Received signal (batch_size, 2*Nr)
    """
        
    SNR_linear = 10 ** (SNR / 10)
        
    # Split real and imaginary parts
    batch_size = x.size(0)
    K = x.size(1) // 2
    x_real = x[:, :K]  # First K dims (real part)
    x_imag = x[:, K:]  # Last K dims (imaginary part)
        
    # Create complex signal: x_complex shape (batch_size, K)
    x_complex = torch.complex(x_real, x_imag)
        
    # Handle dimension mismatch (K vs Nr)
    if K != Nr:
        if K > Nr:
            # If input dimension is larger than number of receive antennas, truncate
            x_complex = x_complex[:, :Nr]
        else:
            # If input dimension is smaller than number of receive antennas, pad with zeros
            padding = torch.zeros(batch_size, Nr - K, dtype=x_complex.dtype, device=x_complex.device)
            x_complex = torch.cat([x_complex, padding], dim=1)
        
    # Calculate input signal power (should be close to 1 after power_normalize)
    signal_power = torch.mean(torch.abs(x_complex)**2, dim=1, keepdim=True)  # (batch_size, 1)
        
    # Calculate noise variance based on SNR and signal power
    # SNR = Signal_Power / Noise_Power  =>  Noise_Power = Signal_Power / SNR
    noise_variance = signal_power / SNR_linear  # (batch_size, 1)
        
    # Generate complex AWGN noise
    # For complex noise, real and imaginary parts each have variance of noise_variance/2
    n_real = torch.randn(batch_size, Nr, device=x.device) * torch.sqrt(noise_variance / 2)
    n_imag = torch.randn(batch_size, Nr, device=x.device) * torch.sqrt(noise_variance / 2)
    n = torch.complex(n_real, n_imag)
        
    # AWGN channel transmission: y = x + n (no channel matrix)
    y_complex = x_complex + n
        
    # Split complex output into real and imaginary parts format (2*Nr dimensions)
    y_real = torch.real(y_complex)  # Shape: (batch_size, Nr)
    y_imag = torch.imag(y_complex)  # Shape: (batch_size, Nr)
        
    # Concatenate real and imaginary parts: (batch_size, 2*Nr)
    y = torch.cat([y_real, y_imag], dim=1)

    return y, noise_variance
