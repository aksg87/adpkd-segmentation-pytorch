import numpy as np

BACKGROUND = 0.0
L_KIDNEY = 0.5019608
R_KIDNEY = 0.7490196

BACKGROUND_INT = 0
L_KIDNEY_INT = 128
R_KIDNEY_INT = 191


class SingleChannelMaskNumpy:
    """Sets 1 for kidneys, 0 otherwise."""

    def __call__(self, label):
        """
        Args:
            label, (1, H, W) uint8 numpy array
        Returns:
            numpy array, (1, H, W) uint8 one-hot encoded mask
        """
        kidney = np.bitwise_or(label == R_KIDNEY_INT, label == L_KIDNEY_INT)
        return kidney.astype(np.uint8)


class TwoChannelsMaskNumpy:
    """
    The first channel for right kidney vs background,
    and the second one for left kidney vs background.

    Kidneys are marked as 1, background as 0.
    """

    def __call__(self, label):
        """
        Args:
            label, (1, H, W) uint8 numpy array
        Returns:
            numpy array, (2, H, W) uint8 one-hot encoded mask
        """
        r_kidney = label == R_KIDNEY_INT
        l_kidney = label == L_KIDNEY_INT
        mask = np.concatenate([r_kidney, l_kidney], axis=0).astype(np.uint8)
        return mask


class ThreeChannelMaskNumpy:
    """One channel for each of the 3 classes. Background last."""

    def __call__(self, label):
        """
        Args:
            label, (1, H, W) float32 tensor

        Returns:
            numpy array, (3, H, W) uint8 one-hot encoded mask
        """
        background = label == BACKGROUND_INT
        r_kidney = label == R_KIDNEY_INT
        l_kidney = label == L_KIDNEY_INT
        mask = np.concatenate([r_kidney, l_kidney, background], axis=0).astype(
            np.uint8
        )
        return mask
