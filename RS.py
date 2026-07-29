import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_multiview_data(I, J, M, N):
    """
    Generates a synthetic 4D multi-view image array (I, J, M, N).
    I, J : Spatial sampling points (camera grid on RS plane)
    M, N : Angular resolution (pixels per elemental direction image)
    """
    print("Generating synthetic multi-view light field array...")
    multi_view = np.zeros((I, J, M, N), dtype=float)
    
    # Create a simple test pattern: a object at depth with parallax
    for i in range(I):
        for j in range(J):
            # Shift center of pattern according to camera position (parallax)
            shift_x = int((i - I / 2) * 1.5)
            shift_y = int((j - J / 2) * 1.5)
            
            # Draw a circle/square pattern in each elemental view
            cx, cy = M // 2 + shift_x, N // 2 + shift_y
            y, x = np.ogrid[:M, :N]
            mask = (x - cx)**2 + (y - cy)**2 <= (M // 6)**2
            
            multi_view[i, j, mask] = 1.0
            
    return multi_view

def ray_to_wavefront_conversion(multi_view_array, use_random_phase=True):
    """
    Converts 4D Ray-Sampling array into a continuous 2D complex wavefront on the RS plane.
    Formula: U_rs(i, j) = 2D_FFT { Ray_Intensity(i, j, theta_x, theta_y) * Phase }
    """
    I, J, M, N = multi_view_array.shape
    total_ny = I * M
    total_nx = J * N
    
    wavefront_rs = np.zeros((total_ny, total_nx), dtype=complex)
    
    for i in range(I):
        for j in range(J):
            # Extract direction image at sampling point (i, j)
            directional_rays = multi_view_array[i, j, :, :]
            
            # Apply diffuse/random phase to avoid severe interference speckle
            if use_random_phase:
                phase_mask = np.exp(1j * np.random.uniform(0, 2 * np.pi, (M, N)))
                complex_rays = np.sqrt(directional_rays) * phase_mask
            else:
                complex_rays = np.sqrt(directional_rays).astype(complex)
            
            # Fourier transform converts ray angular direction into spatial frequency field
            sub_wavefront = np.fft.fftshift(np.fft.fft2(complex_rays))
            
            # Map sub-wavefront into the unified Ray-Sampling plane matrix
            y_start, y_end = i * M, (i + 1) * M
            x_start, x_end = j * N, (j + 1) * N
            wavefront_rs[y_start:y_end, x_start:x_end] = sub_wavefront
            
    return wavefront_rs

def angular_spectrum_propagation(u_in, wavelength, dx, dy, distance):
    """
    Propagates a complex wave field from RS plane to CGH plane via Angular Spectrum Method (ASM).
    """
    Ny, Nx = u_in.shape
    
    # Spatial frequency axes
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    
    # Wavenumber k and Transfer function H
    k = 2 * np.pi / wavelength
    kz_squared = k**2 - (2 * np.pi * FX)**2 - (2 * np.pi * FY)**2
    
    # Evanescent wave filter (real propagating waves only)
    kz = np.sqrt(np.maximum(0, kz_squared))
    H = np.exp(1j * kz * distance)
    
    # Propagate field in Fourier domain
    U_freq = np.fft.fft2(u_in)
    U_propagated = np.fft.ifft2(U_freq * H)
    
    return U_propagated

# ==============================================================================
# Simulation Pipeline Execution
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Parameter Definitions ---
    wavelength = 633e-9      # He-Ne Red Laser (633 nm)
    dx = dy = 8e-6           # Pixel size on RS/Hologram plane (8 micrometers)
    z_distance = 0.05        # Distance from RS plane to Hologram plane (5 cm)
    
    # Array grid setup (I x J camera views, each having M x N angular resolution)
    I_views, J_views = 8, 8   # 8x8 Elemental camera array grid
    M_res, N_res = 32, 32     # 32x32 pixels per direction image
    
    Total_Y = I_views * M_res # 256 pixels
    Total_X = J_views * N_res # 256 pixels

    print(f"RS Plane Grid Resolution: {Total_Y} x {Total_X} pixels")

    # --- 2. Generate Multi-View Array (Ray Capture) ---
    ray_array = generate_synthetic_multiview_data(I_views, J_views, M_res, N_res)

    # --- 3. Convert Rays to Wavefront on RS Plane ---
    wavefront_rs = ray_to_wavefront_conversion(ray_array, use_random_phase=True)

    # --- 4. Propagate Wavefront from RS Plane to Hologram Plane ---
    wavefront_cgh = angular_spectrum_propagation(wavefront_rs, wavelength, dx, dy, z_distance)

    # Extract Phase Mask for Spatial Light Modulator (SLM)
    cgh_phase_hologram = np.angle(wavefront_cgh) % (2 * np.pi)

    # Reconstruct back from hologram plane to check focus quality
    reconstructed_field = angular_spectrum_propagation(wavefront_cgh, wavelength, dx, dy, -z_distance)
    reconstructed_intensity = np.abs(reconstructed_field)**2

    # --- 5. Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Center elemental view
    axes[0, 0].imshow(ray_array[I_views//2, J_views//2], cmap='gray')
    axes[0, 0].set_title("Single Elemental Image (Ray Angles)")
    
    # Converted Amplitude on RS Plane
    axes[0, 1].imshow(np.abs(wavefront_rs), cmap='magma')
    axes[0, 1].set_title("Converted Wavefront Amplitude (RS Plane)")
    
    # Phase-Only Hologram pattern (for SLM display)
    axes[1, 0].imshow(cgh_phase_hologram, cmap='twilight')
    axes[1, 0].set_title("Calculated Hologram Phase Mask [0 - 2π]")
    
    # Numerically Reconstructed Intensity
    axes[1, 1].imshow(reconstructed_intensity, cmap='gray')
    axes[1, 1].set_title("Reconstructed Optical Field Intensity")
    
    for ax in axes.ravel():
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()