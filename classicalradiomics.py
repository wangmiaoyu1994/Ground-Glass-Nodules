import logging
from os import path
from glob import glob
from copy import deepcopy
from pickle import loads, dumps
from zlib import compress, decompress
from concurrent.futures import ProcessPoolExecutor

from radiomics.featureextractor import RadiomicsFeatureExtractor

from imageprocessor import ImageProcessor


class RadiomicsExtractor:
    def __init__(self):
        settings = dict()
        settings["featureClass"] = {ky: list() for ky in ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]}
        settings["setting"] = {"additionalInfo": False}

        self.extractor = RadiomicsFeatureExtractor(deepcopy(settings))
        self.extractor.enableFeatureClassByName("shape")

        self.extractor_wos = RadiomicsFeatureExtractor(deepcopy(settings))

    def mainf(self, prompt, rdf, update=False):
        rf = rdf.replace(".radiodata.pkf", ".radiomics.pkf")

        has_rf = path.exists(rf)

        if has_rf and not update:
            print(prompt, "Been Processed")

            return None

        print(prompt, "Processing")

        with open(rdf, "rb") as rpkf:
            mvol, mseg = loads(decompress(rpkf.read()))

        if has_rf:
            with open(rf, "rb") as rpkf:
                radiomics = loads(decompress(rpkf.read()))
        else:
            radiomics = dict()

        ischanged = False
        for vky in mvol.keys():
            if radiomics.get(vky) is None:
                radiomics[vky] = dict()

            for sky in mseg.keys():
                if radiomics[vky].get(sky) is None:
                    radiomics[vky][sky] = self.extract_radiomics(mvol[vky], mseg[sky], vky == "original")

                    ischanged = True

        if ischanged:
            with open(rf, "wb") as wpkf:
                wpkf.write(compress(dumps(radiomics)))

        print(prompt, "Done")

    def extract_radiomics(self, vol, seg, shape_included=False):
        if shape_included:
            temp = self.extractor.execute(vol, seg)
        else:
            temp = self.extractor_wos.execute(vol, seg)

        subradiomics = {ky.replace("original_", ""): float(temp[ky])
                        for ky in temp.keys()}

        return subradiomics


logging.getLogger("radiomics").setLevel(logging.ERROR)

if __name__ == "__main__":
    srcs = [r"C:\Users\301PBDC\Desktop\dataset\images",
            r"C:\Users\301PBDC\Desktop\intraclass_correlation\images"]

    rst = r"I:\results"

    processor = ImageProcessor()
    extractor = RadiomicsExtractor()

    error_check = []
    with ProcessPoolExecutor(max_workers=9) as executor:
        for src in srcs:
            cdir = path.dirname(src)
            midpath = path.basename(cdir)
            tardir = f"{rst}\\{midpath}"

            rdfs = [f for folder in processor.return_dir(tardir)
                    for f in glob(f"{folder}\\*.radiodata.pkf")]
            nrdf = len(rdfs)

            for i, f in enumerate(rdfs):
                prmt = f"{i + 1} in {nrdf} ({(i + 1) / nrdf:.2%}) {f}"
                error_check.append([prmt, executor.submit(extractor.mainf, prmt, f, update=False)])

    for prmt, ck in error_check:
        try:
            ck.result()
        except Exception:
            print(prmt)
