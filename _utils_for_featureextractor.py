import logging
from itertools import product

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter, generate_binary_structure, binary_propagation

import SimpleITK as sitk
from gudhi.cubical_complex import CubicalComplex
from gtda.homology import CubicalPersistence
from classicalradiomics.featureextractor import RadiomicsFeatureExtractor

global_topo_binwidth = 10


def _pv_and_cr(vol_ary, seg_ary, threshod):
    pv = np.count_nonzero((vol_ary > threshod) * seg_ary)

    return pv, pv / np.count_nonzero(seg_ary)


def _transform_diagram(diagram, fill_num):
    def _subdiagrams(x, dim):
        n_features_in_dim = np.count_nonzero(x[0, :, 2] == dim)
        xs = x[x[:, :, 2] == dim].reshape(x.shape[0], n_features_in_dim, 3)

        return xs[:, :, :2]

    dimensions = np.unique(diagram[:, :, 2])

    subdiagrams = [_subdiagrams(diagram, d) for d in dimensions]
    nfeature = [subdiagram.shape[1] for subdiagram in subdiagrams]

    transdiagram = np.zeros(shape=(diagram.shape[0], dimensions.shape[0], max(nfeature), 2))

    for i, subdiagram in enumerate(subdiagrams):
        transdiagram[:, i, :nfeature[i], :] = subdiagram

    lifetime = transdiagram[:, :, :, 1] - transdiagram[:, :, :, 0]
    trivial_points = lifetime == 0
    transdiagram[trivial_points] = fill_num

    return cp.array(transdiagram), cp.array(lifetime)


def _check_sum_nonzero(x, return_sum=False):
    if return_sum:
        sum_ = x.sum(axis=2)
        sum_[sum_ == 0] = 1

        return sum_
    else:
        x_ = x.copy()
        x_[x_.sum(axis=2) == 0] = 1
        return x_


def _bottleneck(lifetime):
    return lifetime.max(axis=2) / 2


def _wasserstein(lifetime):
    return cp.linalg.norm(lifetime / 2, axis=2, ord=2)


def _perentropy(lifetime):
    w = lifetime / _check_sum_nonzero(lifetime, return_sum=True)[:, :, None]

    w[w == 0] = 1
    w *= - cp.log(w)

    return w.sum(axis=2)


def _points(lifetime):
    w = cp.sign(lifetime)

    return cp.log10(w.sum(axis=2) + 1)


def _bfunc(diagrams, sampling):
    born = sampling >= diagrams[:, :, 0]
    not_dead = sampling < diagrams[:, :, 1]
    alive = cp.logical_and(born, not_dead)
    betti = alive.sum(axis=2).T

    return betti


def _betti(transdiagram, sampling):
    w = cp.concatenate([_bfunc(sd, sampling)[None] for sd in transdiagram])

    return cp.linalg.norm(w, axis=2, ord=2)


def _lfunc(diagrams, sampling):
    n_points = diagrams.shape[1]
    midpoints = (diagrams[:, :, 1] + diagrams[:, :, 0]) / 2.
    heights = (diagrams[:, :, 1] - diagrams[:, :, 0]) / 2.
    fibers = cp.maximum(-cp.abs(sampling - midpoints) + heights, 0)
    top_pos = range(-min(1, n_points), 0)
    fibers.partition(top_pos, axis=2)
    fibers = cp.flip(fibers[:, :, -1:], axis=2)
    fibers = cp.transpose(fibers, (1, 2, 0))
    pad_with = ((0, 0), (0, max(0, 1 - n_points)), (0, 0))
    fibers = cp.pad(fibers, pad_with, "constant", constant_values=0)
    return fibers


def _landscape(transdiagram, sampling):
    w = cp.concatenate([_lfunc(sd, sampling)[None, :, 0, :] for sd in transdiagram])

    return cp.linalg.norm(w, axis=2, ord=2)


def _sfunc(diagrams, sampling):
    sampling = cp.transpose(sampling, axes=(1, 2, 0))
    weights = cp.diff(diagrams, axis=2)
    total_weights = weights.sum(axis=1)
    total_weights[total_weights == 0.] = cp.inf
    midpoints = (diagrams[:, :, [1]] + diagrams[:, :, [0]]) / 2.
    heights = (diagrams[:, :, [1]] - diagrams[:, :, [0]]) / 2.
    fibers = cp.maximum(-cp.abs(sampling - midpoints) + heights, 0)
    fibers_weighted_sum = cp.sum(weights * fibers, axis=1) / total_weights
    return fibers_weighted_sum


def _silhouette(transdiagram, sampling):
    w = cp.concatenate([_sfunc(sd, sampling)[None] for sd in transdiagram])

    return cp.linalg.norm(w, axis=2, ord=2)


def _pfunc(diagrams, sampling):
    p = cp.zeros((len(diagrams), len(sampling), len(sampling)), dtype=float)

    diagrams[:, :, 1] -= diagrams[:, :, 0]

    first_sampling = sampling[0]

    for i, diagram in enumerate(diagrams):
        nontrivial_points_idx = cp.flatnonzero(diagram[:, 1])
        diagram_nontrivial_pixel_coords = (diagram - first_sampling)[nontrivial_points_idx]

        unique, counts = np.unique(cp.asnumpy(diagram_nontrivial_pixel_coords), axis=0, return_counts=True)
        unique = tuple(tuple(row) for row in unique.astype(np.int32).T)
        p[i][unique] = counts
        p[i] = gaussian_filter(p[i], 0.5, mode="constant") * cp.sign(p[i])

    p = cp.transpose(p, axes=(0, 2, 1))

    return p


def _perimage(transdiagram, ranging):
    transdiagram = transdiagram.copy()

    w = cp.concatenate([_pfunc(sd, ranging)[None] for sd in transdiagram]).reshape(*transdiagram.shape[:2], -1)

    return cp.linalg.norm(w, axis=2, ord=2)


def _coef_sfunc(x):
    xt = cp.zeros(2)
    x = x[x[:, 0] != x[:, 1]]

    alpha = cp.linalg.norm(x, axis=1)
    alpha = cp.where(alpha == 0, cp.ones(x.shape[0]), alpha)
    roots = cp.multiply(cp.multiply((x[:, 0] + 1j * x[:, 1]), (x[:, 1] - x[:, 0])), 1. / (cp.sqrt(2) * alpha))

    coefficients = (cp.poly(roots) if roots.any() else cp.array([1.0]))[1:]

    dimension = min(1, coefficients.shape[0])
    xt[:dimension] = coefficients[:dimension].real
    xt[1:1 + dimension] = coefficients[:dimension].imag

    return cp.sign(xt) * cp.log10(cp.abs(xt) + 1)


def _coef_s(transdiagram):
    return cp.concatenate([cp.concatenate([_coef_sfunc(dim_sd)
                                           for dim_sd in sd])[None]
                           for sd in transdiagram])


class TopoMapper:
    def __init__(self, binwidth=global_topo_binwidth, ndim=3):
        self.dims = [0, 1, 2]

        self.binwidth = binwidth

        self.struct = generate_binary_structure(ndim, ndim)

    def get_topomap_image(self, vol, vol_ary, seg_ary):
        vol_ary = vol_ary // self.binwidth * self.binwidth

        max_val = vol_ary.max(initial=0.0, where=seg_ary == 1)
        minpmax = vol_ary.min(initial=0.0, where=seg_ary == 1) + max_val

        trans_ary = {"pos": np.where(seg_ary == 1, vol_ary, max_val + 1),
                     "neg": np.where(seg_ary == 1, minpmax - vol_ary, max_val + 1)}

        topomaps = dict()

        for bky in trans_ary.keys():
            trans_ary_ = trans_ary[bky]

            cuper = CubicalComplex(top_dimensional_cells=trans_ary_)

            cuper.compute_persistence()
            seedbags = cuper.cofaces_of_persistence_pairs()[0]
            nseedbag = len(seedbags)

            trans_ary_ = cp.array(trans_ary_)
            flattened_ary = trans_ary_.flatten(order="F")

            for dim in self.dims:
                betti_map = cp.zeros_like(trans_ary_)
                lifetime_map = cp.zeros_like(trans_ary_, dtype=cp.int32)
                silhouette_map = cp.zeros_like(trans_ary_, dtype=cp.float64)

                if dim < nseedbag:
                    if len(seedbags[dim]) != 0:
                        for bi, di in seedbags[dim]:
                            b, d = flattened_ary[[bi, di]]

                            seed = cp.zeros_like(flattened_ary)
                            seed[bi] = seed[di] = 1

                            fmask = trans_ary_ < d
                            tmask = binary_propagation(seed.reshape(trans_ary_.shape, order="F"), self.struct, fmask)
                            pmask = tmask * (trans_ary_ >= b)

                            betti_map += pmask

                            lifetime = d - b
                            lifetime_map += tmask * lifetime
                            silhouette_map += cp.minimum(trans_ary_ - b, d - trans_ary_) * pmask * lifetime

                    silhouette_map /= cp.maximum(lifetime_map, 1)

                betti_vol = sitk.GetImageFromArray(cp.asnumpy(betti_map))
                silhouette_vol = sitk.GetImageFromArray(cp.asnumpy(silhouette_map))

                betti_vol.CopyInformation(vol)
                silhouette_vol.CopyInformation(vol)

                topomaps[f"topomap-{bky}-betti-H{dim}"] = betti_vol
                topomaps[f"topomap-{bky}-silhouette-H{dim}"] = silhouette_vol

        return topomaps


class Perputor:
    def __init__(self, binwidth=global_topo_binwidth):
        self.cuper = CubicalPersistence(homology_dimensions=(0, 1, 2))

        self.binwidth = binwidth

    def compute_persistence(self, vol_ary, seg_ary):
        vol_ary = vol_ary // self.binwidth * self.binwidth

        max_val = vol_ary.max(initial=0.0, where=seg_ary == 1)
        minpmax = vol_ary.min(initial=0.0, where=seg_ary == 1) + max_val

        diagram = self.cuper.fit_transform([np.where(seg_ary == 1, vol_ary, max_val + 1),
                                            np.where(seg_ary == 1, minpmax - vol_ary, max_val + 1)])

        diagram[1, :, :2] = minpmax - diagram[1, :, 1::-1]

        return diagram


class RadiomicExtractor:
    def __init__(self):
        settings = dict()
        settings["featureClass"] = {ky: list() for ky in ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]}
        settings["setting"] = {"additionalInfo": False}

        self.fextractor = RadiomicsFeatureExtractor(settings)
        self.fextractor.enableFeatureClassByName("shape")

        self.pextractor = RadiomicsFeatureExtractor(settings)

    def extract(self, vol, seg, compute_shape_features=False):
        if compute_shape_features:
            temp_features = self.fextractor.execute(vol, seg)
        else:
            temp_features = self.pextractor.execute(vol, seg)

        return {ky.replace("original_", "radiomic_"): float(temp_features[ky])
                for ky in temp_features.keys()}


class QuantoExtractor:
    @staticmethod
    def extract(vol_ary, seg_ary, max_axis, min_axis, mid_axis):
        quanto_features = dict()
        quanto_features["SCV_N135"], quanto_features["CTR_N135"] = _pv_and_cr(vol_ary, seg_ary, -135)
        quanto_features["SCV_N250"], quanto_features["CTR_N250"] = _pv_and_cr(vol_ary, seg_ary, -250)

        temp_1 = min_axis / (max_axis + min_axis)
        temp_2 = max_axis * min_axis + mid_axis ** 2

        quanto_features["ACO"] = 2 * temp_1
        quanto_features["AEL"] = max_axis * min_axis / temp_2 - temp_1
        quanto_features["AFL"] = mid_axis ** 2 / temp_2 - temp_1
        quanto_features["KEL"] = 1 - mid_axis / max_axis
        quanto_features["KFL"] = 1 - min_axis / mid_axis
        quanto_features["MPS"] = (min_axis ** 2 / (mid_axis * max_axis)) ** (1 / 3)

        return {f"quanto_{ky}": float(quanto_features[ky])
                for ky in quanto_features.keys()}


class ToporadiomicExtractor:
    def __init__(self):
        settings = dict()
        settings["featureClass"] = {ky: list() for ky in ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]}
        settings["setting"] = {"additionalInfo": False}

        self.bextractor = RadiomicsFeatureExtractor(settings)
        self.bextractor.settings["binWidth"] = 3

        self.sextractor = RadiomicsFeatureExtractor(settings)
        self.sextractor.settings["binWidth"] = 10

    def extract(self, topomaps, seg):
        toporadiomic = dict()
        for tmky in topomaps.keys():
            if "betti" in tmky:
                temp_features = self.bextractor.execute(topomaps[tmky], seg)
            else:
                temp_features = self.sextractor.execute(topomaps[tmky], seg)

            toporadiomic.update({f"{tmky}_{ky}": float(temp_features[ky])
                                 for ky in temp_features.keys()})

        return toporadiomic


class TopoanalyticExtractor:
    def __init__(self, binwidth=global_topo_binwidth, ndim=3, polycoef=1):
        self.bindx = {"pos": 0, "neg": 1}

        self.hindx = {f"H{indx}": indx
                      for indx in range(ndim)}

        self.cindx = {f"{hiky}-{cky}{cindx}": self.hindx[hiky] * 2 * polycoef + cindx
                      for hiky, cky, cindx in product(self.hindx.keys(), ("R", "I"), range(polycoef))}

        self.cuper = CubicalPersistence(homology_dimensions=(0, 1, 2))

        self.calculators_a = {"bottleneck": _bottleneck,
                              "wasserstein": _wasserstein,
                              "perentropy": _perentropy,
                              "points": _points}

        self.calculators_b = {"betti": _betti,
                              "landscape": _landscape,
                              "silhouette": _silhouette}

        self.calculators_c = {"perimage": _perimage}

        self.calculators_d = {"coef-S": _coef_s}

        self.binwidth = binwidth

    def extract(self, vol_ary, seg_ary, diagram):
        vol_ary = vol_ary[seg_ary == 1] // self.binwidth * self.binwidth

        min_val = vol_ary.min()
        max_val = vol_ary.max()

        sampling = cp.array(np.unique(vol_ary))[:, None, None]

        ranging_base = cp.arange(min_val, max_val + 1, self.binwidth)[:, None]
        ranging = cp.concatenate([ranging_base, ranging_base + 1 - min_val], axis=1)

        transdiagram, lifetime = _transform_diagram(diagram, max_val + 1)

        temp = {ky: self.calculators_a[ky](lifetime)
                for ky in self.calculators_a.keys()}

        temp.update({ky: self.calculators_b[ky](transdiagram, sampling)
                     for ky in self.calculators_b.keys()})

        temp.update({ky: self.calculators_c[ky](transdiagram, ranging)
                     for ky in self.calculators_c.keys()})

        temp.update({ky: self.calculators_d[ky](transdiagram)
                     for ky in self.calculators_d.keys()})

        tfkys = [ky for ky in temp.keys() if "coef" not in ky]
        analytic = {f"topodescriptor-{b}_{f}_{h}": float(temp[f][self.bindx[b], self.hindx[h]])
                    for b, f, h in product(self.bindx.keys(), tfkys, self.hindx.keys())}

        tfkys = [ky for ky in temp.keys() if "coef" in ky]
        analytic.update({f"topodescriptor-{b}_{f}_{c}": float(temp[f][self.bindx[b], self.cindx[c]])
                         for b, f, c in product(self.bindx.keys(), tfkys, self.cindx.keys())})

        return analytic


logging.getLogger("radiomics").setLevel(logging.ERROR)
