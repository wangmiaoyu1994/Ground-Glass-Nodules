from os import path
from glob import glob
from pickle import loads, dumps
from zlib import compress, decompress
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import generate_binary_structure as np_structure, binary_propagation as np_propagation
import cupy as cp
from cupyx.scipy.ndimage import generate_binary_structure as cp_structure, binary_propagation as cp_propagation

import SimpleITK as sitk
from gudhi.cubical_complex import CubicalComplex

from imageprocessor import ImageProcessor


class TopoMapper:
    def __init__(self, ndim=3):
        self.bindex = {"pos": 0, "neg": 1}
        self.dims = [d for d in range(ndim)]

        self.binwidth = 10

        self.np_structure = np_structure(ndim, ndim)
        self.cp_structure = cp_structure(ndim, ndim)

    def mainf(self, prompt, rdf, use_cpu, update):
        tmapf = rdf.replace(".radiodata.pkf", ".topomap.pkf")

        has_tmapf = path.exists(tmapf)

        if has_tmapf and not update:
            print(prompt, "Been Processed")

            return None

        print(prompt, "Processing")

        with open(rdf, "rb") as rpkf:
            mvol, mseg = loads(decompress(rpkf.read()))

        seg_ary = sitk.GetArrayFromImage(mseg["supra"])

        if has_tmapf:
            with open(tmapf, "rb") as rpkf:
                tmap = loads(decompress(rpkf.read()))
        else:
            tmap = dict()

        ischanged = False
        for vky in mvol.keys():
            if vky != "original":
                continue

            if tmap.get(vky) is None:
                tmap[vky] = dict()

                vol = mvol[vky]

                vol_ary = np.int32(sitk.GetArrayFromImage(vol) // self.binwidth * self.binwidth)

                max_val = vol_ary.max(initial=0.0, where=seg_ary == 1)
                minpmax = vol_ary.min(initial=0.0, where=seg_ary == 1) + max_val

                packed_ary = [np.where(seg_ary == 1, vol_ary, max_val + 1),
                              np.where(seg_ary == 1, minpmax - vol_ary, max_val + 1)]

                tmap[vky] = self.generate_topomap(packed_ary, vol, use_cpu)

                ischanged = True

        if ischanged:
            with open(tmapf, "wb") as wpkf:
                wpkf.write(compress(dumps(tmap)))

        print(prompt, "Done")

    def generate_topomap(self, packed_ary, ref_vol, use_cpu=True):
        subtmap = dict()

        if use_cpu:
            _xp = np
            _structure = self.np_structure
            _propagation = np_propagation
        else:
            _xp = cp
            _structure = self.cp_structure
            _propagation = cp_propagation

        for bky in self.bindex.keys():
            ary = packed_ary[self.bindex[bky]]

            cuper = CubicalComplex(top_dimensional_cells=ary)

            cuper.compute_persistence()
            seedbags = cuper.cofaces_of_persistence_pairs()[0]
            nseedbag = len(seedbags)

            ary = ary if use_cpu else cp.array(ary)
            flattened_ary = ary.flatten(order="F")

            for dim in self.dims:
                total_lifetime = 0

                betti_map = _xp.zeros_like(ary)
                landscape_map = _xp.zeros_like(ary)
                silhouette_map = _xp.zeros_like(ary, dtype=_xp.float64)

                if dim < nseedbag:
                    if len(seedbags[dim]) != 0:
                        for bi, di in seedbags[dim]:
                            b, d = flattened_ary[[bi, di]]

                            seed = _xp.zeros_like(flattened_ary)
                            seed[bi] = seed[di] = 1

                            rmask = (ary >= b) & (ary < d)
                            pmask = _propagation(seed.reshape(ary.shape, order="F"), _structure, rmask)

                            betti_map += pmask

                            landscape = _xp.minimum(ary - b, d - ary) * pmask
                            landscape_map = _xp.maximum(landscape_map, landscape)

                            lifetime = d - b
                            total_lifetime += lifetime
                            silhouette_map += landscape * lifetime

                    silhouette_map /= total_lifetime

                if use_cpu:
                    betti_vol = sitk.GetImageFromArray(betti_map)
                    landscape_vol = sitk.GetImageFromArray(landscape_map)
                    silhouette_vol = sitk.GetImageFromArray(silhouette_map)
                else:
                    betti_vol = sitk.GetImageFromArray(cp.asnumpy(betti_map))
                    landscape_vol = sitk.GetImageFromArray(cp.asnumpy(landscape_map))
                    silhouette_vol = sitk.GetImageFromArray(cp.asnumpy(silhouette_map))

                betti_vol.CopyInformation(ref_vol)
                landscape_vol.CopyInformation(ref_vol)
                silhouette_vol.CopyInformation(ref_vol)

                subtmap[f"topomap-{bky}-betti-H{dim}"] = betti_vol
                subtmap[f"topomap-{bky}-landscape-H{dim}"] = landscape_vol
                subtmap[f"topomap-{bky}-silhouette-H{dim}"] = silhouette_vol

        return subtmap


if __name__ == "__main__":
    srcs = [r"C:\Users\301PBDC\Desktop\dataset\images",
            r"C:\Users\301PBDC\Desktop\intraclass_correlation\images"]

    rst = r"I:\results"

    processor = ImageProcessor()
    mapper = TopoMapper()

    error_check = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        for src in srcs:
            cdir = path.dirname(src)
            midpath = path.basename(cdir)
            tardir = f"{rst}\\{midpath}"

            rdfs = [f for folder in processor.return_dir(tardir)
                    for f in glob(f"{folder}\\*.radiodata.pkf")]
            nrdf = len(rdfs)

            for i, rf in enumerate(rdfs):
                prmt = f"{i + 1} in {nrdf} ({(i + 1) / nrdf:.2%}) {rf}"
                error_check.append([prmt, executor.submit(mapper.mainf, prmt, rf, use_cpu=i % 3 != 0, update=True)])

    for prmt, ck in error_check:
        try:
            ck.result()
        except Exception:
            print(prmt)
