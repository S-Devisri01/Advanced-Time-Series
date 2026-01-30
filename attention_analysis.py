import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

def analyze_attention_patterns(model, X_sample, feature_names, timesteps_to_show=24):
    """Extract and visualize attention patterns"""

    print("Extracting attention weights from the model...")

    # Find the attention layer
    attention_layer = None
    for layer in model.layers:
        if 'multi_head_attention' in layer.name:
            attention_layer = layer
            break

    if attention_layer is None:
        print(" Could not find attention layer in the model")
        return None

    # Create a model that outputs attention weights
    try:
        # Try different approaches for attention extraction
        attention_model = Model(
            inputs=model.input,
            outputs=attention_layer.output
        )

        # Get attention output
        sample_idx = 0
        X_sample_single = X_sample[sample_idx:sample_idx+1]
        attention_output = attention_model.predict(X_sample_single, verbose=0)

        # Handle different output structures
        if isinstance(attention_output, list):
            # Newer TF versions return list
            attention_weights = attention_output[1] if len(attention_output) > 1 else attention_output[0]
        else:
            # Older TF versions
            attention_weights = attention_output

        print(f"Attention weights shape: {attention_weights.shape}")

        # Average across heads and samples
        avg_attention = np.mean(attention_weights, axis=1)[0]

        # Visualize attention matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Full attention matrix
        im1 = axes[0].imshow(avg_attention, cmap='viridis', aspect='auto')
        axes[0].set_title('Attention Weight Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Key Timestep')
        axes[0].set_ylabel('Query Timestep')
        plt.colorbar(im1, ax=axes[0])

        # Zoom into last 24x24 timesteps
        if avg_attention.shape[0] >= 24:
            zoom_matrix = avg_attention[-24:, -24:]
            im2 = axes[1].imshow(zoom_matrix, cmap='viridis', aspect='auto')
            axes[1].set_title('Attention (Last 24 Hours)', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Key Timestep (relative)')
            axes[1].set_ylabel('Query Timestep (relative)')
            plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig('attention_weights.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Create attention summary
        print("\n ATTENTION FOCUS SUMMARY:")
        print("-" * 80)

        # Calculate attention distribution
        total_attention = np.sum(avg_attention)

        # Attention to self
        self_attention = np.diag(avg_attention)
        self_percentage = np.sum(self_attention) / total_attention * 100
        print(f"Attention to same timestep: {self_percentage:.1f}%")

        # Attention to recent (last 6 hours)
        recent_mask = np.zeros_like(avg_attention)
        for i in range(avg_attention.shape[0]):
            recent_mask[i, max(0, i-6):i] = 1
        recent_percentage = np.sum(avg_attention * recent_mask) / total_attention * 100
        print(f"Attention to recent 6 hours: {recent_percentage:.1f}%")

        # Attention to daily patterns (same hour, previous 7 days)
        daily_mask = np.zeros_like(avg_attention)
        for i in range(avg_attention.shape[0]):
            for j in range(1, 8):
                idx = i - 24*j
                if idx >= 0:
                    daily_mask[i, idx] = 1
        daily_percentage = np.sum(avg_attention * daily_mask) / total_attention * 100
        print(f"Attention to same hour previous days: {daily_percentage:.1f}%")

        return avg_attention

    except Exception as e:
        print(f" Error extracting attention weights: {e}")
        print("Note: This might be due to TensorFlow version differences.")
        print("Continuing with other analyses...")
        return None