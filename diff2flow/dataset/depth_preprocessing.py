import torch
import warnings
import numpy as np


MAX_FAR_PLANE = 120


def distance_to_planar_depth(distance, focal_length: float = 886.81, height: int = 768, width: int = 1024):
    """
    Convert distance to focal point to planar depth. Adapted from
    https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697

    Args:
        distance: distance to focal point (depth_meters.hdf5), shape (h, w)
        focal_length: focal length of camera
        height, width: height and width of image
    Returns:
        planar_depth: planar depth, shape (h, w)
    """
    npyImageplaneX = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width).reshape(1, width).repeat(height, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height).reshape(height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([height, width, 1], focal_length, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)
    planar_depth = distance / np.linalg.norm(npyImageplane, 2, 2) * focal_length
    return planar_depth


def interpolate_nans(array, fill_values=1e-4):
    """
    Interpolate nans in a depth image by using the closest non-nan value.
    """
    nans = np.isnan(array)
    if not nans.any():
        return array
    x = lambda z: z.nonzero()[0]
    array[nans] = np.interp(x(nans), x(~nans), array[~nans], left=fill_values, right=fill_values)
    assert not np.isnan(array).any(), "There are still nans in the array."
    return array


def calculate_valid_mask(depth, dataset_name):
    if dataset_name == "hypersim":
        q_min, q_max = [1e-4, min(MAX_FAR_PLANE, 1e9)]
    elif dataset_name == "vkitti2":
        q_min, q_max = [0, min(MAX_FAR_PLANE, 655)]
    elif dataset_name == "diode":
        q_min, q_max = [0.1, 120]
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    if isinstance(depth, np.ndarray):
        mask = np.logical_and(depth >= q_min, depth <= q_max)
    elif isinstance(depth, torch.Tensor):
        mask = torch.logical_and(depth >= q_min, depth <= q_max)
    else:
        raise ValueError(f"Invalid depth type {type(depth)}")
    
    return mask[None]


def resize(sample, size):
    if len(sample.shape) == 4:
        return torch.nn.functional.interpolate(sample, size=size)
    elif len(sample.shape) == 3:
        if sample.shape[0] > 3:
            # H x W x C -> C x H x W
            sample = sample.permute(2, 0, 1)
        return torch.nn.functional.interpolate(sample[None], size=size).squeeze(0)
    elif len(sample.shape) == 2:
        return torch.nn.functional.interpolate(sample[None, None], size=size).squeeze(0).squeeze(0)
    else:
        raise ValueError("Invalid shape: {}".format(sample.shape))


def preprocess_depth(depth, dataset_name, out_channels=3, keep_raw_depth=False):
    """
    Args:
        depth: depth map in numpy format and range [0, inf] w. shape (H, W)
        dataset_name: name of the dataset (e.g. vkitti2, hypersim)
    Returns:
        depth: normalized depth map with shape (C, H, W) and range [-1, 1]
        valid_mask: mask of valid depth values with shape (1, H, W)
    """
    # 2. calculate valid mask
    valid_mask = calculate_valid_mask(depth, dataset_name)

    # 1. cap depth to the far plane
    depth = np.minimum(depth, MAX_FAR_PLANE)

    # 3. interpolate nans
    depth = interpolate_nans(depth)

    # 4. normalize depth
    if dataset_name == 'hypersim':
        depth = distance_to_planar_depth(depth)

    if keep_raw_depth:
        depth = depth[None].repeat(out_channels, axis=0)
        return depth, valid_mask
    
    q_min, q_max = np.percentile(depth, (1, 99))
    q_min = np.log(q_min)
    q_max = np.log(q_max)
    depth = np.log(depth)
    depth = (depth - q_min) / (q_max - q_min)
    depth = (depth - 0.5) * 2

    # 5. add channel dimension
    depth = depth[None].repeat(out_channels, axis=0)
    
    return depth, valid_mask


class DatasetPreprocessor:
    def __init__(
            self,
            size=None,
            depth_key="depth",
            out_channels=3,
            return_valid_mask=True,
            exclude_keys_for_resize=None,
            keep_raw_depth=False,
        ):
        self.size = tuple(size) if size is not None else None
        self.out_channels = out_channels
        self.return_valid_mask = return_valid_mask
        self.exclude_keys_for_resize = exclude_keys_for_resize or []
        self.saved_processed_sample = None
        self.keep_raw_depth = keep_raw_depth
        self.depth_key = depth_key

    def preprocess_sample(self, sample):
        """ get dataset name """
        if "dataset" not in sample:
            sample["dataset"] = "hypersim" # default to hypersim
        dataset_name = sample.get("dataset")
        assert isinstance(self.size, tuple) or isinstance(self.size, list) or self.size is None, "Invalid size"
        try:
            dataset_name = dataset_name.decode() # convert bytes to string
        except AttributeError:
            pass
        sample["dataset"] = dataset_name

        """ exceptions """
        # if dataset_name in ["depth_anything"]:
        #   ...

        """ Preprocess depth map """
        depth = sample[self.depth_key]
        depth, valid_mask = preprocess_depth(depth, dataset_name, self.out_channels, self.keep_raw_depth)
        
        if self.return_valid_mask:
            if "valid_mask" in sample:
                valid_mask_sample = sample["valid_mask"]
                # merge the valid masks
                valid_mask = valid_mask * valid_mask_sample
            sample["valid_mask"] = valid_mask
        sample[self.depth_key] = depth

        """ convert to tensor and resize """
        if self.size is not None:
            for key in sample:
                if isinstance(sample[key], np.ndarray):
                    sample[key] = torch.tensor(sample[key], dtype=torch.float32)
                if key in self.exclude_keys_for_resize:
                    continue
                if isinstance(sample[key], torch.Tensor):
                    sample[key] = resize(sample[key], size=self.size)

            # filter the resized valid mask with 1
            if self.return_valid_mask:
                sample["valid_mask"] = sample["valid_mask"] == 1

        return sample
    
    def __call__(self, sample):
        try:
            processed_sample = self.preprocess_sample(sample)
            self.saved_processed_sample = processed_sample
            return processed_sample
        except Exception as e:
            print("Error in preprocessing a sample")
            print(e)
            if self.saved_processed_sample is not None:
                return self.saved_processed_sample
            raise e


def unnormalize_depth(depth, q_min, q_max, normalization_fn: str = 'log'):
    """
    Unnormalize depth map from [-1, 1] to actual depth values.
    Args:
        depth: Torch tensor depth map in range [-1, 1].
        q_min: Min quantile used for normalization (planar depths).
        q_max: Max quantile used for normalization (planar depths).
        normalization_fn: Function used for normalization (log,
            inverse, identity).
    Returns:
        Unnormalized depth map in range [0, inf].
    """
    # First of all, unnormalize from [-1,1] to [0,1]
    depth = depth / 2 + 0.5

    # Then, unnormalize from [0,1] using the corresponding quantile
    if normalization_fn == "log":
        q_min = torch.log(torch.tensor(q_min))
        q_max = torch.log(torch.tensor(q_max))
    elif normalization_fn == "inverse":
        warnings.warn("Inverse normalization is not working well yet!")
        q_min = 1.0 / q_min
        q_max = 1.0 / q_max
    depth = depth * (q_max - q_min) + q_min

    # Finally, unnormalize using the function
    if normalization_fn == "identity":
        return depth
    elif normalization_fn == "log":
        return torch.exp(depth)
    elif normalization_fn == "inverse":
        depth = depth.clamp(1e-6, 1e6)
        return 1.0 / depth
    else:
        raise ValueError(f"Unknown normalization function: {normalization_fn}")
