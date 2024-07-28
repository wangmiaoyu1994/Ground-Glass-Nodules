import logging
from os import path
from glob import glob
from copy import deepcopy
from pickle import loads, dumps
from zlib import compress, decompress
from concurrent.futures import ProcessPoolExecutor

from radiomics.featureextractor import RadiomicsFeatureExtractor
from radiomics.imageoperations import getWaveletImage, getLoGImage

from imageprocessor import ImageProcessor


class RadiomicsExtractor:
    def __init__(self):
        settings = dict()
        settings["featureClass"] = {ky: list() for ky in ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]}
        settings["setting"] = {"additionalInfo": False}

        self.extractor_nt = RadiomicsFeatureExtractor(deepcopy(settings))
        self.extractor_nt.settings["binWidth"] = 0.5

        self.extractor_wt = RadiomicsFeatureExtractor(deepcopy(settings))
        self.extractor_wt.settings["binWidth"] = 5

    @staticmethod
    def islabeled(tky):
        return all(ky not in tky for ky in ["original", "wavelet", "log-sigma"])

    def mainf(self, prompt, tmapf, update=False):
        trf = tmapf.replace(".topomap.pkf", ".toporadiomics.pkf")

        has_trf = path.exists(trf)

        if has_trf and not update:
            print(prompt, "Been Processed")

            return None

        print(prompt, "Processing")

        rdf = tmapf.replace(".topomap.pkf", ".radiodata.pkf")
        with open(rdf, "rb") as rpkf:
            mvol, mseg = loads(decompress(rpkf.read()))

        with open(tmapf, "rb") as rpkf:
            tmap = loads(decompress(rpkf.read()))

        ischanged = False
        for vky in tmap.keys():
            if all("wavelet" not in tky for tky in tmap[vky].keys()):
                tmap[vky].update({f"{tky}_{cls}": vol
                                  for tky in tmap[vky].keys()
                                  if "log-sigma" not in tky and "original" not in tky
                                  for vol, cls, _ in getWaveletImage(tmap[vky][tky], None)
                                  if cls == "wavelet-HHH"})

                ischanged = True

            if all("log-sigma" not in tky for tky in tmap[vky].keys()):
                tmap[vky].update({f"{tky}_{cls}": vol
                                  for tky in tmap[vky].keys()
                                  if "wavelet" not in tky and "original" not in tky
                                  for vol, cls, _ in getLoGImage(tmap[vky][tky], None, sigma=[0.25])})

                ischanged = True

            if all("original" not in tky for tky in tmap[vky].keys()):
                tmap[vky] = {tky if self.islabeled(tky) else f"{tky}_original": tmap[vky][tky]
                             for tky in tmap[vky].keys()}

                ischanged = True

        if ischanged:
            with open(tmapf, "wb") as wpkf:
                wpkf.write(compress(dumps(tmap)))

        if has_trf:
            with open(trf, "rb") as rpkf:
                tradiomics = loads(decompress(rpkf.read()))
        else:
            tradiomics = dict()

        ischanged = False
        for vky in tmap.keys():
            if vky != "original":
                continue

            for tky in tmap[vky].keys():
                if all(ky not in tky for ky in ["original", "wavelet-HHH", "log-sigma-0-25"]):
                    continue

                if tradiomics.get(f"{vky}_{tky}") is None:
                    tradiomics[f"{vky}_{tky}"] = dict()

                for sky in mseg.keys():
                    if tradiomics[f"{vky}_{tky}"].get(sky) is None:
                        tradiomics[f"{vky}_{tky}"][sky] = self.extract_tradiomics(tmap[vky][tky], mseg[sky], tky)

                        ischanged = True

        if ischanged:
            with open(trf, "wb") as wpkf:
                wpkf.write(compress(dumps(tradiomics)))

        print(prompt, "Done")

    def extract_tradiomics(self, vol, seg, image_class):
        if "silhouette" in image_class:
            temp = self.extractor_nt.execute(vol, seg)
        else:
            temp = self.extractor_wt.execute(vol, seg)

        subtradiomcis = {ky.replace("original_", ""): float(temp[ky])
                         for ky in temp.keys()}

        return subtradiomcis


logging.getLogger("radiomics").setLevel(logging.ERROR)

if __name__ == "__main__":
    srcs = [r"C:\Users\301PBDC\Desktop\dataset\images",
            r"C:\Users\301PBDC\Desktop\intraclass_correlation\images"]

    rst = r"I:\results"

    processor = ImageProcessor()
    extractor = RadiomicsExtractor()

    error_check = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        for src in srcs:
            cdir = path.dirname(src)
            midpath = path.basename(cdir)
            tardir = f"{rst}\\{midpath}"

            tmapfs = [f for folder in processor.return_dir(tardir)
                      for f in glob(f"{folder}\\*.topomap.pkf")]
            ntmapf = len(tmapfs)

            for i, tmf in enumerate(tmapfs):
                prmt = f"{i + 1} in {ntmapf} ({(i + 1) / ntmapf:.2%}) {tmf}"
                error_check.append([prmt, executor.submit(extractor.mainf, prmt, tmf, update=True)])

    for prmt, ck in error_check:
        try:
            ck.result()
        except Exception:
            print(prmt)
