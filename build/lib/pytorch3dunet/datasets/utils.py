import collections
import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('Dataset')


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def prediction_collate(cls, batch):
        """Default collate_fn. Override in child class for non-standard datasets."""
        return default_prediction_collate(batch)


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(self, raw_datasets, label_datasets, weight_dataset, patch_shape, stride_shape, **kwargs):
        """
        :param raw_datasets: ndarray of raw data
        :param label_datasets: ndarray of ground truth labels
        :param weight_dataset: ndarray of weights for the labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param kwargs: additional metadata
        """

        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_datasets[0], patch_shape, stride_shape)
        if label_datasets is None:
            self._label_slices = None
        else:
            # take the first element in the label_datasets to build slices
            self._label_slices = self._build_slices(label_datasets[0], patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset[0], patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'


class FilterSliceBuilder(SliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label
    """

    def __init__(self, raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index=(0,),
                 threshold=0.6, slack_acceptance=0.01, **kwargs):
        super().__init__(raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, **kwargs)
        if label_datasets is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = np.copy(label_datasets[0][label_idx])
            for ii in ignore_index:
                patch[patch == ii] = 0
            non_ignore_counts = np.count_nonzero(patch != 0)
            non_ignore_counts = non_ignore_counts / patch.size
            return non_ignore_counts > threshold or rand_state.rand() < slack_acceptance

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


class EmbeddingsSliceBuilder(FilterSliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label and patches containing more than
    `patch_max_instances` labels
    """

    def __init__(self, raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index=(0,),
                 threshold=0.8, slack_acceptance=0.01, patch_max_instances=48, patch_min_instances=5, **kwargs):
        super().__init__(raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index,
                         threshold, slack_acceptance, **kwargs)

        if label_datasets is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = label_datasets[0][label_idx]
            num_instances = np.unique(patch).size

            # patch_max_instances is a hard constraint
            if num_instances <= patch_max_instances:
                # make sure that we have at least patch_min_instances in the batch and allow some slack
                return num_instances >= patch_min_instances or rand_state.rand() < slack_acceptance

            return False

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


class RandomFilterSliceBuilder(EmbeddingsSliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label and return only random sample of those.
    """

    def __init__(self, raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index=(0,),
                 threshold=0.8, slack_acceptance=0.01, patch_max_instances=48, patch_acceptance_probab=0.1,
                 max_num_patches=25, **kwargs):
        super().__init__(raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape,
                         ignore_index=ignore_index, threshold=threshold, slack_acceptance=slack_acceptance,
                         patch_max_instances=patch_max_instances, **kwargs)

        self.max_num_patches = max_num_patches

        if label_datasets is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            result = rand_state.rand() < patch_acceptance_probab
            if result:
                self.max_num_patches -= 1

            return result and self.max_num_patches > 0

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)

class AroundCenterSliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(self, raw_datasets, label_datasets, centers, weight_dataset, patch_shape, stride_shape, **kwargs):
        """
        :param raw_datasets: ndarray of raw data
        :param label_datasets: ndarray of ground truth labels
        :param centers: ndarray of slice centers [n,3]
        :param weight_dataset: ndarray of weights for the labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param kwargs: additional metadata
        """

        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_datasets[0], patch_shape, stride_shape, centers)
        
        
        if label_datasets is None:
            self._label_slices = None
        else:
            # take the first element in the label_datasets to build slices
            mask_slices = [(slice(0,label_datasets[0].shape[0]),)+slcs for slcs in self._raw_slices]    
            self._label_slices = mask_slices 
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            raise ## not yet....
            self._weight_slices = self._build_slices(weight_dataset[0], patch_shape, stride_shape, centers)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape, centers):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
                

        slices += AroundCenterSliceBuilder.get_slices_around_centers((i_z, i_y, i_x), centers[0][0], patch_shape)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'
    

    ## MY CODE
    @staticmethod
    def _get_slice_around_center(image_shape, center, patch_shape, centers_set, margin=[8,12,12]):
        """
        image_shape (x,y,z)
        center (x,y,z)
        patch_shape (x,y,z)
        margin (x_m,y_m,z_m)
        get slice around center randomly(by margin)
        
        """
        x,y,z = image_shape
        x_m,y_m,z_m = margin
        cen_x, cen_y, cen_z = center
        p_x,p_y,p_z = [(p//2, p-(p//2)) for p in patch_shape]
        
        cen_x += np.random.randint(-x_m, x_m+1)
        cen_y += np.random.randint(-y_m, y_m+1)
        cen_z += np.random.randint(-z_m, z_m+1)
        
        cen_x = p_x[0] if cen_x-p_x[0]<0 else x-p_x[1] if cen_x+p_x[1]>x else cen_x
        cen_y = p_y[0] if cen_y-p_y[0]<0 else y-p_y[1] if cen_y+p_y[1]>y else cen_y
        cen_z = p_z[0] if cen_z-p_z[0]<0 else z-p_z[1] if cen_z+p_z[1]>z else cen_z
        
        if (cen_x,cen_y,cen_z) in centers_set:
            return None
        else:
            centers_set.add((cen_x,cen_y,cen_z))
        
        
        x = (cen_x-p_x[0], cen_x+p_x[1])
        y = (cen_y-p_y[0], cen_y+p_y[1])
        z = (cen_z-p_z[0], cen_z+p_z[1])
        
        slc = (
            slice(*x),
            slice(*y),
            slice(*z)
        )
        return slc

    @staticmethod
    def get_slices_around_centers(image_shape, centers, patch_shape, slices_per_center=20, margin=[8,12,12]):
        slices = list()
        centers_set = set()
        for center in centers:
            for _ in range(slices_per_center):
                slc = AroundCenterSliceBuilder._get_slice_around_center(image_shape, center, patch_shape, centers_set, margin)
                if slc:
                    slices.append(slc)
        return slices     

def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')


def _loader_classes(class_name):
    modules = [
        'pytorch3dunet.datasets.hdf5',
        'pytorch3dunet.datasets.dsb',
        'pytorch3dunet.datasets.utils'
    ]
    return get_class(class_name, modules)


def get_slice_builder(raws, labels, weight_maps, config, centers=None):
    assert 'name' in config
    logger.info(f"Slice builder config: {config}")
    slice_builder_cls = _loader_classes(config['name'])
    if centers:
        return slice_builder_cls(raws, labels, centers, weight_maps, **config)
    else:
        return slice_builder_cls(raws, labels, weight_maps, **config)


def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warn(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
        "Train and validation 'file_paths' overlap. One cannot use validation data for training!"

    train_datasets = dataset_class.create_datasets(loaders_config, phase='train')

    val_datasets = dataset_class.create_datasets(loaders_config, phase='val')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }


def get_test_loaders(config):
    """
    Returns test DataLoader.

    :return: generator of DataLoader objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating test set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warn(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    test_datasets = dataset_class.create_datasets(loaders_config, phase='test')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        logger.info(f'Loading test set from: {test_dataset.file_path}...')
        if hasattr(test_dataset, 'prediction_collate'):
            collate_fn = test_dataset.prediction_collate
        else:
            collate_fn = default_prediction_collate

        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                         collate_fn=collate_fn)


def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img.ravel() for img in images]
    )
    return np.min(flat), np.max(flat), np.mean(flat), np.std(flat)


def sample_instances(label_img, instance_ratio, random_state, ignore_labels=(0,)):
    """
    Given the labelled volume `label_img`, this function takes a random subset of object instances specified by `instance_ratio`
    and zeros out the remaining labels.

    Args:
        label_img(nd.array): labelled image
        instance_ratio(float): a number from (0, 1]
        random_state: RNG state
        ignore_labels: labels to be ignored during sampling

    Returns:
         labelled volume of the same size as `label_img` with a random subset of object instances.
    """
    unique = np.unique(label_img)
    for il in ignore_labels:
        unique = np.setdiff1d(unique, il)

    # shuffle labels
    random_state.shuffle(unique)
    # pick instance_ratio objects
    num_objects = round(instance_ratio * len(unique))
    if num_objects == 0:
        # if there are no objects left, just return an empty patch
        return np.zeros_like(label_img)

    # sample the labels
    sampled_instances = unique[:num_objects]

    result = np.zeros_like(label_img)
    # keep only the sampled_instances
    for si in sampled_instances:
        result[label_img == si] = si

    return result
