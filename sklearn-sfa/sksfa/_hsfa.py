import numpy as np
from sksfa.utils import ReceptiveRebuilder, ReceptiveSlicer
from sklearn.preprocessing import PolynomialFeatures
from sksfa import SFA
from time import time

class Flatten:
    def fit(self, X, y=None):
        pass

    def partial(self, X, y=None):
        pass

    def transform(self, X):
        return X.reshape((X.shape[0], -1))

class AdditiveNoise:
    def __init__(self, std=0.01):
        self.std = std

    def fit(self, X, y=None):
        pass

    def partial(self, X, y=None):
        pass

    def transform(self, X):
        return X + np.random.normal(0.0, self.std, X.shape)

class Clipper:
    def __init__(self, val_min=-4, val_max=+4):
        self.val_min = val_min
        self.val_max = val_max

    def fit(self, X, y=None):
        pass

    def partial(self, X, y=None):
        pass

    def transform(self, X):
        return np.clip(X, self.val_min, self.val_max)


class HSFA:
    """Hierarchical Slow Feature Analysis (HSFA).

    A network of SFA estimators interlaced with receptive field transformers
    and linear SFA estimators for intermediate pre-expansion dimensionality reduction.
    This can deal with high-dimensional image time-series significantly better than
    standard (non-linear) SFA by using receptive fields to slice the images in a way
    comparable to convolutional layers in neural networks.

    In each layer, the image representation is first sliced into receptive fields that
    are defined by field-dimensions and corresponding strides. The field inputs are then
    fed as flattened input-batches to a combination of linear SFA (for dimensionality
    reduction), quadratic polynomial expansion, and linear SFA (for feature extraction).
    The dimension after the reduction is the same as the number of subsequently extracted
    features and can be specified for each layer individually.
    The final layer does not need to be specified and always consists of the same combination,
    but without prior slicing into receptive fields.

    Important: SFA estimators cannot be further fit after using them to transform input. This
    is why for training an HSFA network, the data has to be repeatedly fed through the network
    until the last layer is trained. HSFA's 'fit' function will take care of the logistics of
    that.

    ----------
    n_components : int
        Number of features extracted by the complete network.
    final_degree : int, default=2
        The degree of the final layer's polynomial expansion.
    input_shape : tuple (int)
        The shape of a single input (i.e., without sample-dimension) to the
        input layer.
    layer_configurations : list of 6-tuples
        A list of tuples to configure the intermediate layers. Each tuple needs to contain:
        (field_width, field_height, stride_width, stride_height, n_intermediate_components, polynomial_degree)
    internal_batch_size : int, default=50
        The size of mini-batches used internally. This should not be chosen too small as
        the SFA nodes at this point do not respect connections between batches.
    noise_std : float, default=0.05
        Additive noise added at intermediate layers. Crank this up if you run into problems
        with singular covariance matrices during training. This noise will not be applied
        at transformation time.
        In general, this has a slight regularizing effect, but should not be chosen too high.
        If you run into repeated problems, consider changing your network architecture and/or
        increase the size of your dataset.
    verbose : int, default=0
        Whether to print additional information. 0 means no output, 1 means output during training,
        2 means output during training and also during transformation

    Attributes
    ----------
    layer_configurations : list of tuples
        This contains all layer configurations except the final, fully-connected one.
    input_shape : tuple
        See 'input_shape' parameter.
    internal_batch_size : int
        See 'internal_batch_size' parameter.
    n_components : int
        The number of output features.
    sequence : list of transformers/estimators
        This list will contain the used transformers and estimators in correct order.
    layer_outputs : list
        This list will contain all the output shapes of all intermediate layers.
    self.

    Examples
    --------
    >>> from sksfa import HSFA
    >>> import numpy as np
    >>> n_samples = 5000
    >>> image_width, image_height = 10, 10
    >>> dimension = image_width * image_height
    >>> t = np.linspace(0, 8*np.pi, n_samples).reshape(n_samples, 1)
    >>> t = t * np.arange(1, dimension + 1)
    >>>
    >>> ordered_cosines = np.cos(t)
    >>> mixed_cosines = np.dot(ordered_cosines, np.random.normal(0, 1, (dimension, dimension)))
    >>> mixed_cosines = mixed_cosines.reshape(n_samples, image_width, image_height, 1)
    >>> layer_configurations = [(5, 5, 5, 5, 4, 1)]
    >>>
    >>> hsfa = HSFA(2, mixed_cosines.shape[1:], layer_configurations, noise_std=0.1)
    >>> hsfa = hsfa.fit(mixed_cosines)
    >>> unmixed_cosines = hsfa.transform(mixed_cosines)
   """
    def __init__(self, n_components, input_shape, layer_configurations, final_degree=2, internal_batch_size=50, noise_std=0.05, verbose=False):
        self.layer_configurations = layer_configurations
        self.verbose = verbose
        self.input_shape = input_shape
        self.internal_batch_size = internal_batch_size
        self.n_components = n_components
        self.noise_std = noise_std
        self.sequence = []
        self.layer_outputs = []
        self.final_degree = final_degree
        self.initialize_layers()

    def initialize_layers(self):
        # Stack all layers except the last
        for build_idx, (field_w, field_h, stride_w, stride_h, n_components, poly_degree) in enumerate(self.layer_configurations):
            if build_idx > 0 and (field_w == field_h == -1):
                field_w = slicer.reconstruction_shape[0]
                field_h = slicer.reconstruction_shape[1]
            try:
                input_shape = self.input_shape if build_idx == 0 else slicer.reconstruction_shape
                slicer = ReceptiveSlicer(input_shape=input_shape, field_size=(field_w, field_h), strides=(stride_w, stride_h))
            except AssertionError:
                raise ValueError(f"Layer {build_idx + 1}: Field ({field_w}, {field_h}) with stride ({stride_w}, {stride_h}) does not fit data dimension ({input_shape[0]}, {input_shape[1]})")
            self.sequence.append(slicer)
            if poly_degree > 1:
                pre_expansion_sfa = SFA(n_components, batch_size=self.internal_batch_size, fill_mode=None)
                self.sequence.append(pre_expansion_sfa)
                expansion = PolynomialFeatures(poly_degree)
                expansion.partial = expansion.fit
                self.sequence.append(expansion)
            self.sequence.append(AdditiveNoise(self.noise_std))
            post_expansion_sfa = SFA(n_components, batch_size=self.internal_batch_size, fill_mode=None)
            self.sequence.append(post_expansion_sfa)
            self.sequence.append(Clipper(-4, 4))
            reconstructor = ReceptiveRebuilder((slicer.reconstruction_shape))
            self.sequence.append(reconstructor)
            self.layer_outputs.append(slicer.reconstruction_shape)
            if self.verbose > 0:
                print(f"WxH output layer {build_idx + 1}: " + str(slicer.reconstruction_shape))
        # Flatten
        self.sequence.append(Flatten())
        # Last layer
        if self.final_degree > 1:
            pre_expansion_sfa = SFA(self.n_components, batch_size=self.internal_batch_size, fill_mode=None)
            self.sequence.append(pre_expansion_sfa)
            expansion = PolynomialFeatures(self.final_degree)
            expansion.partial = expansion.fit
            self.sequence.append(expansion)
        self.sequence.append(AdditiveNoise(self.noise_std))
        post_expansion_sfa = SFA(self.n_components, batch_size=self.internal_batch_size, fill_mode=None)
        self.sequence.append(post_expansion_sfa)
        self.sequence.append(Clipper(-4, 4))
        if self.verbose > 0:
            print("Shape of final output: " + str((self.n_components,)))

    def fit(self, X):
        X = np.copy(X)
        n_samples = X.shape[0]
        batch_size = self.internal_batch_size
        n_batches = int(np.ceil(n_samples / batch_size))
        accumulating_indices = [idx for idx, member in enumerate(self.sequence) if type(member) == SFA]
        last_idx = -1
        for i, idx in enumerate(accumulating_indices):
            # Already trained part of sequence:
            transform_only = self.sequence[:last_idx+1]
            # Part of sequence to train:
            partial_sequence = self.sequence[last_idx+1:idx+1]
            
            # This whole block is only for verbose printout:
            if self.verbose > 0:
                receptive_rebuilder_positions = [type(e) == ReceptiveRebuilder for e in self.sequence[:idx]]
                current_layer = 1 + sum(receptive_rebuilder_positions) # count number of receptive rebuilders up to now
                id_last_receptive_rebuilder = max(np.where(receptive_rebuilder_positions)[0]) if any(receptive_rebuilder_positions) else 0
                num_sfa = 1 + sum([type(e) == SFA for e in self.sequence[id_last_receptive_rebuilder:idx]]) # count number of SFAs since last receptive rebuilder
                print(f"Training layer {current_layer}, SFA {num_sfa} ({i+1} of {len(accumulating_indices)} total)")
                try:
                    from tqdm import tqdm
                except ImportError:
                    raise ImportError("For verbose output, the tqdm package needs to be installed")
                batch_iterator = tqdm(range(n_batches), desc="Processed batches", unit="batches")
            else:
                batch_iterator = range(n_batches)
            
            for batch_idx in batch_iterator:
                current_batch = X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                for member in transform_only:
                    current_batch = member.transform(current_batch)
                for member in partial_sequence:
                    member.partial(current_batch)
                    if type(member) is not SFA:
                        current_batch = member.transform(current_batch)
            last_idx = idx
        return self

    def transform(self, X, seq_end=None):
        n_samples = X.shape[0]
        batch_size = self.internal_batch_size
        n_batches = int(np.ceil(n_samples / batch_size))
        result = None
        sequence = self.sequence if seq_end is None else self.sequence[:seq_end]
        if self.verbose == 2:
            try:
                from tqdm import tqdm
            except ImportError:
                raise ImportError("For verbose output, the tqdm package needs to be installed")
            iterator = tqdm(range(n_batches), desc="Transformed batches", unit="batches")
        else:
            iterator = range(n_batches)
        for batch_idx in iterator:
            current_batch = X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            for transformer in sequence:
                if type(transformer) == AdditiveNoise:
                    continue
                current_batch = transformer.transform(current_batch)
            if result is None:
                result = np.empty((n_samples,) + current_batch.shape[1:])
            result[batch_idx * batch_size: (batch_idx + 1) * batch_size] = current_batch
        return result

    def summary(self):
        """ Prints a summary of the network architecture.
        """
        print("\n = = = = NETWORK ARCHITECTURE = = = = \n")
        print("Input Layer:")
        print(f"\tinput shape: \t\t{self.input_shape}")
        for layer_idx, (field_w, field_h, stride_w, stride_h, n_components, poly_degree) in enumerate(self.layer_configurations):
            print(f"Layer {layer_idx + 1}:")
            print(f"\treceptive field: \t({field_w}, {field_h})\n\tstrides: \t\t({stride_w}, {stride_h})\n\texpansion degree: \t{poly_degree}")
            output_shape = self.layer_outputs[layer_idx]
            print(f"\toutput shape: \t\t{output_shape + (n_components,)}")
        print(f"Final Layer:")
        print("\tfully connected")
        print(f"\texpansion degree \t{self.final_degree}")
        print(f"\toutput shape \t\t({self.n_components},)\n\n")


