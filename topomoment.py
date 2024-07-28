from os import path
from glob import glob
from pickle import loads, dumps
from zlib import compress, decompress
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import SimpleITK as sitk

from imageprocessor import ImageProcessor


class TopoMomentCalculator:
    def mainf(self, prompt, rdf, tmapf, update=False):
        tmomf = rdf.replace(".radiodata.pkf", ".topomoment.pkf")

        has_tmomf = path.exists(tmomf)

        if has_tmomf and not update:
            print(prompt, "Been Processed")

            return None

        print(prompt, "Processing")

        with open(rdf, "rb") as rpkf:
            mvol, mseg = loads(decompress(rpkf.read()))

        mvry = {vky: sitk.GetArrayFromImage(mvol[vky]) for vky in mvol.keys()}
        msry = {sky: sitk.GetArrayFromImage(mseg[sky]) for sky in mseg.keys()}

        with open(tmapf, "rb") as rpkf:
            tmap = loads(decompress(rpkf.read()))

        tmry = {vky: {tmky: sitk.GetArrayFromImage(tmap[vky][tmky])
                      for tmky in tmap[vky].keys()
                      if "wavelet" not in tmky and "log-sigma" not in tmky}
                for vky in tmap.keys()}

        if has_tmomf:
            with open(tmomf, "rb") as rpkf:
                tmoment = loads(decompress(rpkf.read()))
        else:
            tmoment = dict()

        ischanged = False
        for vky in mvry.keys():
            if vky != "original":
                continue

            if tmoment.get(vky) is None:
                tmoment[vky] = dict()

            for sky in msry.keys():
                if tmoment[vky].get(sky) is None:
                    tmoment[vky][sky] = dict()

                for tmky in tmry[vky].keys():
                    if all(tmky not in ky for ky in tmoment[vky][sky].keys()):
                        temp = self.calculate_moment(mvry[vky], tmry[vky][tmky], msry[sky])

                        tmoment[vky][sky].update({f"{tmky}_{fky}": float(temp[fky])
                                                  for fky in temp.keys()})

                        ischanged = True

        if ischanged:
            with open(tmomf, "wb") as wpkf:
                wpkf.write(compress(dumps(tmoment)))

        print(prompt, "Done")

    @staticmethod
    def calculate_moment(x, w, m=None):
        if m is not None:
            w = np.where(m == 1, w, 0)

        if np.sum(w) == 0:
            w[:, :, :] = 1

        subtmoment = dict()

        subtmoment["mean"] = np.average(x, weights=w)

        subtmoment["variacne"] = np.average((x - subtmoment["mean"]) ** 2, weights=w)

        std = np.sqrt(subtmoment["variacne"])

        subtmoment["skew"] = np.average(((x - subtmoment["mean"]) / std) ** 3, weights=w) if std != 0 else 0

        subtmoment["kurtosis"] = np.average(((x - subtmoment["mean"]) / std) ** 4, weights=w) if std != 0 else -3

        return subtmoment


if __name__ == "__main__":
    srcs = [r"C:\Users\301PBDC\Desktop\dataset\images",
            r"C:\Users\301PBDC\Desktop\intraclass_correlation\images"]

    rst = r"I:\results"

    processor = ImageProcessor()
    calculator = TopoMomentCalculator()

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

                tmf = rf.replace(".radiodata.pkf", ".topomap.pkf")

                error_check.append([prmt, executor.submit(calculator.mainf, prmt, rf, tmf, update=True)])

    for prmt, ck in error_check:
        try:
            ck.result()
        except Exception:
            print(prmt)
