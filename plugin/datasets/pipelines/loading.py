import os
import mmcv
import numpy as np
from PIL import Image
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module(force=True)
class LoadMultiViewImagesFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filenames']
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [i.astype(np.float32) for i in img]
        results['img'] = img
        results['img_shape'] = [i.shape for i in img]
        results['ori_shape'] = [i.shape for i in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [i.shape for i in img]
        # results['scale_factor'] = 1.0
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__} (to_float32={self.to_float32}, '\
            f"color_type='{self.color_type}')"


@PIPELINES.register_module(force=True)
class LoadAID4ADSatelliteImage(object):
    """Load satellite image from AID4AD dataset."""

    def __init__(self, sat_root, img_size=(200, 100), normalize=True):
        self.img_size = img_size
        self.normalize = normalize
        self.token2file = {}
        for loc in os.listdir(sat_root):
            loc_dir = os.path.join(sat_root, loc)
            if not os.path.isdir(loc_dir):
                continue
            for fname in os.listdir(loc_dir):
                if not fname.endswith('.png'):
                    continue
                token = fname.split('_', 1)[1].replace('.png', '')
                self.token2file[token] = os.path.join(loc_dir, fname)
        print(f'LoadAID4ADSatelliteImage: indexed {len(self.token2file)} frames')

    def __call__(self, results):
        token = results['token']
        if token in self.token2file:
            img = Image.open(self.token2file[token]).convert('RGB')
            img = img.resize(self.img_size, Image.LANCZOS)
            img = np.array(img, dtype=np.float32)
            if self.normalize:
                img = img / 255.0
            img = img.transpose(2, 0, 1)
            results['sat_img'] = img
        else:
            h, w = self.img_size[1], self.img_size[0]
            results['sat_img'] = np.zeros((3, h, w), dtype=np.float32)
        return results
