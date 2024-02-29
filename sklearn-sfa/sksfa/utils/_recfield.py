"""
This module contains the core implementations needed to use receptive fields.
"""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from scipy.sparse import issparse

class ReceptiveRebuilder(TransformerMixin, BaseEstimator):
    """ Reconstruction part of field slicing

    This transformer takes input of shape (n_field_samples, n_features) and, given
    a reconstruction shape reshapes it to (n_samples, width, height, n_features) by
    simply reshaping.
    This is necessary to reconstruct the between-field structure in a sample to
    re-apply the slicer.
    
    Parameters
    ----------
    reconstruction_shape : tuple
        A tuple defining the local structure to reconstruct without the
        sample dimension, e.g., (8, 8) will result the output to be
        of shape (n_samples, 8, 8, n_features).

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    Examples
    --------
    >>> from sksfa.utils import ReceptiveRebuilder
    >>> import numpy as np
    >>>
    >>> # This could come out of a slicer + transformation.
    >>> sliced_input = np.repeat(np.arange(9)[..., None], 4, axis=1)
    >>> print(f"Input shape: {sliced_input.shape}")
    Input shape: (9, 4)
    >>> for idx, sample in enumerate(sliced_input): print(f"Sample {idx}: {sample}")
    Sample 0: [0 0 0 0]
    Sample 1: [1 1 1 1]
    Sample 2: [2 2 2 2]
    Sample 3: [3 3 3 3]
    Sample 4: [4 4 4 4]
    Sample 5: [5 5 5 5]
    Sample 6: [6 6 6 6]
    Sample 7: [7 7 7 7]
    Sample 8: [8 8 8 8]
    >>> rebuilder = ReceptiveRebuilder(reconstruction_shape=(3, 3))
    >>> rebuilder = rebuilder.fit(sliced_input)
    >>>
    >>> output = rebuilder.transform(sliced_input)
    >>> print(f"Output shape: {output.shape}")
    Output shape: (1, 3, 3, 4)
    >>> print("Output sample:")
    Output sample:
    >>> for channel_idx in range(4): print(f"Channel {channel_idx}:\\n{output[..., channel_idx].squeeze()}")
    Channel 0:
    [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    Channel 1:
    [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    Channel 2:
    [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    Channel 3:
    [[0. 1. 2.]
     [3. 4. 5.]
     [6. 7. 8.]]
    """
    def __init__(self, reconstruction_shape, copy=True):
        self.reconstruction_shape = reconstruction_shape
        self.input_shape = None
        self.copy = copy
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """Fits the transformer to input X. This mainly checks
        the input and stores the input-shape for dimension
        consistency.

        Parameters
        ----------
        X : {array-like}, shape (n_field_samples, n_features)
            The training input samples.

        y : None or {array-like}, shape (n_samples, 1)
            This does nothing and is only in here for compliance with
            sklearn API.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, dtype=[np.float64, np.float32], copy=self.copy)
        self.input_shape = X.shape[1:]
        self.is_fitted_ = True
        return self

    def partial(self, X, y=None):
        if not self.is_fitted_:
            return self.fit(X)
        else:
            return self

    def transform(self, X):
        """ Applies the reshape transformation to an input stream,
        this should restore the between-field structure of previously sliced
        data, while in-field structure is ignored by keeping it flat.

        Parameters
        ----------
        X : {array-like}, shape (n_field_samples, n_features)
            The field samples to puzzle back together according to
            self.reconstruction_shape.

        Returns
        -------
        X : {array-like}, shape (n_samples,) + reconstruction_shape + (n_features,)
            The samples with restored between-field structure.
        """
        X = check_array(X, dtype=[np.float64, np.float32], copy=self.copy)
        assert(X.shape[1:] == self.input_shape)
        n_features = X.shape[-1]
        original_n_samples = int(np.product(X.shape)/(n_features * np.product(self.reconstruction_shape)))
        n_fields = int(X.shape[0] / original_n_samples)
        output = np.empty((original_n_samples,) + self.reconstruction_shape + (n_features,))
        for sample_idx in range(original_n_samples):
            puzzle_pieces = X[sample_idx::original_n_samples]
            output[sample_idx] = puzzle_pieces.reshape(self.reconstruction_shape + (n_features,))
        return output

class ReceptiveSlicer(TransformerMixin, BaseEstimator):
    """ Slicing part of field slicing.

    This transformer takes input of shape (n_samples, width, height, channels) and slices
    inputs in a receptive field manner.

    Parameters
    ----------
    field_size : tuple
        Shape of the receptive field as a tuple of integers.

    strides : tuple
        Strides in each axis as tuple of integers.

    padding : str
        Either "valid" or "same". Only "valid" is implemented as of now.

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    Examples
    --------
    >>> from sksfa.utils import ReceptiveSlicer
    >>> import numpy as np
    >>>
    >>> ones = np.ones((2, 2))
    >>> # This could be an image or rebuilt output from a lower layer
    >>> data = np.block([[0 * ones, 1 * ones], [2 * ones, 3 * ones]])[None, ..., None]
    >>>
    >>> print(data.squeeze())
    [[0. 0. 1. 1.]
     [0. 0. 1. 1.]
     [2. 2. 3. 3.]
     [2. 2. 3. 3.]]
    >>> print(f"Input shape: {data.shape}")
    Input shape: (1, 4, 4, 1)
    >>> slicer = ReceptiveSlicer(input_shape=data.shape, field_size=ones.shape, strides=(1, 1))
    >>> slicer = slicer.fit(data)
    >>> sliced_output = slicer.transform(data)
    >>> print(f"Output shape: {sliced_output.shape}")
    Output shape: (9, 4)
    >>> for idx, field_sample in enumerate(sliced_output): print(f"Output sample {idx}: {field_sample.squeeze()}")
    Output sample 0: [0. 0. 0. 0.]
    Output sample 1: [0. 1. 0. 1.]
    Output sample 2: [1. 1. 1. 1.]
    Output sample 3: [0. 0. 2. 2.]
    Output sample 4: [0. 1. 2. 3.]
    Output sample 5: [1. 1. 3. 3.]
    Output sample 6: [2. 2. 2. 2.]
    Output sample 7: [2. 3. 2. 3.]
    Output sample 8: [3. 3. 3. 3.]
    """
    def __init__(self, input_shape, field_size=(3, 3), strides=(1, 1), padding="valid", copy=True):
        self.field_size = field_size
        self.input_shape = input_shape
        self.strides = strides
        self.padding = padding
        self.copy = copy
        self.is_fitted_ = False
        self.input_shape = None
        width_steps = self._checkValidSteps(input_shape[0], field_size[0], strides[0])
        height_steps = self._checkValidSteps(input_shape[1], field_size[1], strides[1])
        self.reconstruction_shape = (width_steps, height_steps)

    def fit(self, X, y=None):
        """Fit the model to X. This mainly means checking the input array
        and storing its shape for reconstruction.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, width, height, n_samples)
            The training input samples.

        y : None or {array-like}, shape (n_samples, 1)
            This does nothing and is only in here for compliance with
            sklearn API.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, dtype=[np.float64, np.float32], allow_nd=True,
                        copy=self.copy)
       # self.input_dim_ = X.shape[:1]
        if issparse(X):
            raise TypeError('Slicer does not support sparse input.')
        if self.padding == "valid":
            self._fitValid(X)
        self.is_fitted_ = True
        return self

    def partial(self, X, y=None):
        if not self.is_fitted_:
            return self.fit(X)
        else:
            return self

    def _fitValid(self, X):
        self.input_shape = X.shape[1:]
        n_samples, width, height, channels = X.shape
        field_width, field_height = self.field_size
        width_stride, height_stride = self.strides
        n_steps_width = self._checkValidSteps(width, field_width, width_stride)
        n_steps_height = self._checkValidSteps(height, field_height, height_stride)
        n_output_features = np.product(self.field_size) * channels
        assert(n_steps_width > 0)
        assert(n_steps_height > 0)
        self.reconstruction_shape = (n_steps_width, n_steps_height)

    def _checkValidSteps(self, dimension, field_size, field_stride):
        """ Asserts if splitting up works along a single dimension for a given
        field_size and stride. Returns the number of steps if possible otherwise
        throws an error.

        Parameters
        ----------
        dimension : int
            Size of the dimension to be sliced.

        field_size : int
            Size of the field in this dimension.

        field_stride : int
            Size of the stride in this dimension.

        Returns
        -------
        n_valid_steps : int
            Number of slices, given the provided parameters.
        """
        n_valid_steps = (dimension - field_size)/field_stride + 1
        assert(int(n_valid_steps) == n_valid_steps)
        return int(n_valid_steps)

    def _sliceSingleSample(self, sample, field_rows, field_cols, row_stride, col_stride):
        """ Internal generator that yields slices of a single sample according
        to provided field_size and strides.

        Parameters
        ----------
        X : {array-like}, shape (width, height, channels)
            The samples to be transformed, possibly after padding.

        Yields
        -------
        single_field : ndarray, shape (n_field_samples, field_width * field_height * channels)
            A single field sliced from the input sample. Produces all samples, columns first.
        """
        row_start = 0
        col_start = 0
        while (row_start + field_rows <= sample.shape[0]):
            while (col_start + field_cols <= sample.shape[1]):
                single_field = sample[row_start:row_start + field_rows, col_start:col_start + field_cols, :].flatten()
                yield single_field
                col_start = col_start + col_stride
            col_start = 0
            row_start = row_start + row_stride

    def _transformValid(self, X):
        """ Internal function to perform the slicing with "valid" padding, aka no
        padding at all.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, width, height, channels)
            The samples to be transformed.

        Returns
        -------
        output : ndarray, shape (n_field_samples, field_width * field_height * channels)
            The sliced fields of all samples. The field entries are flattened into
            the last dimension.
        """
        n_samples, width, height, channels = X.shape
        n_steps_width, n_steps_height = self.reconstruction_shape
        n_output_features = np.product(self.field_size) * channels
        self.parts_per_sample = n_steps_width * n_steps_height
        n_output_samples = n_samples * self.parts_per_sample
        output = np.empty((n_output_samples, n_output_features))
        for sample_idx, sample in enumerate(X):
            for part_idx, part in enumerate(self._sliceSingleSample(sample, *self.field_size, *self.strides)):
                output[part_idx * n_samples + sample_idx] = part
        return output

    def transform(self, X):
        """ For a given dataset of images, slice the images into smaller samples in a receptive
        field fashion.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, width, height, channels)
            The samples to be transformed.

        Returns
        -------
        output : ndarray, shape (n_field_samples, field_width * field_height * channels)
            The sliced fields of all samples. The field entries are flattened into
            the last dimension.
        """
        X = check_array(X, dtype=[np.float64, np.float32], copy=self.copy, ensure_2d=False, allow_nd=True)
        assert(X.shape[1:] == self.input_shape)
        output = None
        if self.padding == "valid":
            output = self._transformValid(X)
        return output

if __name__ == "__main__":
    samples = np.ones((20, 9, 9, 1))
    for i in range(samples.shape[0]):
        samples[i] *= i
    sl = ReceptiveSlicer(input_shape=samples.shape[1:], field_size=(4, 4), strides=(1, 1))
    sl.fit(samples)
    hidden = sl.transform(samples)
    sr = ReceptiveRebuilder(reconstruction_shape = sl.reconstruction_shape)
    sr.fit(hidden)
    output = sr.transform(hidden)




