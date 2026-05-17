import random
import cv2
import numpy as np

def remap_brats_labels(label_volume):
    """
    Remap BraTS labels from sparse {0, 1, 2, 4} → contiguous {0, 1, 2, 3}.
    BraTS raw labels:
        0 → background
        1 → NCR/NET (Necrotic Core + Non-enhancing Tumor)
        2 → ED  (Peritumoral Edema)
        4 → ET  (Enhancing Tumor)  ← note: label 3 is unused in BraTS
    """
    remapped = np.zeros_like(label_volume, dtype=np.uint8)
    remapped[label_volume == 1] = 1   # NCR/NET
    remapped[label_volume == 2] = 2   # ED
    remapped[label_volume == 4] = 3   # ET
    return remapped

def normalize_modality(volume):
    """
    Z-score normalize MRI intensities using only non-zero voxels.
    Zero voxels represent regions outside the brain mask (skull-stripped data).
    They are kept at exactly 0.0 so the model can use them as a mask.
    """
    volume = volume.astype(np.float32)
    non_zero_mask = volume != 0

    if not np.any(non_zero_mask):
        return volume   # completely empty scan

    pixels = volume[non_zero_mask]
    mean   = pixels.mean()
    std    = max(pixels.std(), 1e-8)   # prevent division by zero

    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[non_zero_mask] = (volume[non_zero_mask] - mean) / std
    return normalized

class BraTSAugmentation:
    """
    Lightweight CPU data augmentation for BraTS MRI slices.
    """
    def __init__(
        self,
        flip_h_prob    = 0.5,
        flip_v_prob    = 0.3,
        rot90_prob     = 0.4,
        rot_small_prob = 0.3,
        brightness_prob= 0.4,
        noise_prob     = 0.3,
        shift_prob     = 0.3,
        noise_std      = 0.05,
        max_shift      = 0.1,
        max_angle_deg  = 15,
    ):
        self.flip_h_prob     = flip_h_prob
        self.flip_v_prob     = flip_v_prob
        self.rot90_prob      = rot90_prob
        self.rot_small_prob  = rot_small_prob
        self.brightness_prob = brightness_prob
        self.noise_prob      = noise_prob
        self.shift_prob      = shift_prob
        self.noise_std       = noise_std
        self.max_shift       = max_shift
        self.max_angle_deg   = max_angle_deg

    def __call__(self, image, label):
        # 1. Horizontal flip
        if random.random() < self.flip_h_prob:
            image = image[:, :, ::-1].copy()
            label = label[:, ::-1].copy()

        # 2. Vertical flip
        if random.random() < self.flip_v_prob:
            image = image[:, ::-1, :].copy()
            label = label[::-1, :].copy()

        # 3. Random 90° rotation
        if random.random() < self.rot90_prob:
            k = random.randint(1, 3)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            label = np.rot90(label, k=k, axes=(0, 1)).copy()

        # 4. Small angle rotation ±max_angle_deg
        if random.random() < self.rot_small_prob:
            angle  = random.uniform(-self.max_angle_deg, self.max_angle_deg)
            h, w   = label.shape
            center = (w / 2.0, h / 2.0)
            M      = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            rotated_channels = [
                cv2.warpAffine(image[c], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                for c in range(image.shape[0])
            ]
            image = np.stack(rotated_channels, axis=0)
            label = cv2.warpAffine(label.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0).astype(np.int64)

        # 5. Random brightness/contrast adjustment per-channel
        if random.random() < self.brightness_prob:
            for c in range(image.shape[0]):
                alpha = random.uniform(0.8, 1.2)
                beta  = random.uniform(-0.1, 0.1)
                image[c] = image[c] * alpha + beta

        # 6. Additive Gaussian noise
        if random.random() < self.noise_prob:
            noise         = np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
            brain_mask    = (image != 0).astype(np.float32)
            image         = image + noise * brain_mask

        # 7. Per-channel random intensity shift
        if random.random() < self.shift_prob:
            for c in range(image.shape[0]):
                shift     = random.uniform(-self.max_shift, self.max_shift)
                image[c] += shift

        return image.astype(np.float32), label.astype(np.int64)
