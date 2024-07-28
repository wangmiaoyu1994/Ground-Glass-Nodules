from os import path
from glob import glob
from pickle import loads, dumps
from zlib import compress, decompress
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import SimpleITK as sitk
from gtda.homology import CubicalPersistence

from imageprocessor import ImageProcessor


class Persistence:
    def __init__(self):
        self.cuper = CubicalPersistence(homology_dimensions=(0, 1, 2))
        self.binwidth = 10

    def mainf(self, prompt, rdf, update=False):
        pdf = rdf.replace(".radiodata.pkf", ".perdiagram.pkf")

        has_pdf = path.exists(pdf)

        if has_pdf and not update:
            print(prompt, "Been Processed")

            return None

        print(prompt, "Processing")

        with open(rdf, "rb") as rpkf:
            mvol, mseg = loads(decompress(rpkf.read()))

        mvry = {vky: sitk.GetArrayFromImage(mvol[vky]) // self.binwidth * self.binwidth for vky in mvol.keys()}
        msry = {sky: sitk.GetArrayFromImage(mseg[sky]) for sky in mseg.keys()}

        if has_pdf:
            with open(pdf, "rb") as rpkf:
                pdiagram = loads(decompress(rpkf.read()))
        else:
            pdiagram = dict()

        for vky in mvry.keys():
            if all(ky not in vky for ky in ["original", "wavelet-HHH", "log-sigma-0-25"]):
                continue

            if pdiagram.get(vky) is None:
                pdiagram[vky] = dict()

            vol_ary = mvry[vky]
            for sky in msry.keys():
                if pdiagram[vky].get(sky) is None:
                    seg_ary = msry[sky]

                    max_val = vol_ary.max(initial=0.0, where=seg_ary == 1)
                    minpmax = vol_ary.min(initial=0.0, where=seg_ary == 1) + max_val

                    packed_ary = [np.where(seg_ary == 1, vol_ary, max_val + 1),
                                  np.where(seg_ary == 1, minpmax - vol_ary, max_val + 1)]

                    pdiagram[vky][sky] = self.compute_persistence(packed_ary, minpmax)

        with open(pdf, "wb") as wpkf:
            wpkf.write(compress(dumps(pdiagram)))

        print(prompt, "Done")

    def compute_persistence(self, packed_ary, minpmax):
        subpdiagram = self.cuper.fit_transform(packed_ary)

        subpdiagram[1, :, :2] = minpmax - subpdiagram[1, :, 1::-1]

        return subpdiagram


if __name__ == "__main__":
    srcs = [r"C:\Users\301PBDC\Desktop\dataset\images",
            r"C:\Users\301PBDC\Desktop\intraclass_correlation\images"]

    rst = r"I:\results"

    processor = ImageProcessor()
    computor = Persistence()

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
                error_check.append([prmt, executor.submit(computor.mainf, prmt, rf, update=True)])

    for prmt, ck in error_check:
        try:
            ck.result()
        except Exception:
            print(prmt)
