from os import path
from glob import glob
from pickle import loads, dumps
from zlib import compress, decompress
from concurrent.futures import ProcessPoolExecutor

from itertools import product

from gtda.diagrams import Amplitude, PersistenceEntropy, NumberOfPoints, ComplexPolynomial
from imageprocessor import ImageProcessor


class TopoDescriptor:
    def __init__(self, polycoef=3):
        self.bindx = {"pos": 0, "neg": 1}

        self.hindx = {f"H{indx}": indx
                      for indx in [0, 1, 2]}

        self.cindx = {f"{hiky}-{cky}{cindx}": self.hindx[hiky] * 2 * polycoef + cindx
                      for hiky, cky, cindx in product(self.hindx.keys(), ("R", "I"), range(polycoef))}

        self.calculators = {"bottleneck": Amplitude(metric="bottleneck"),
                            "wasserstein": Amplitude(metric="wasserstein"),
                            "betti": Amplitude(metric="betti"),
                            "landscape": Amplitude(metric="landscape"),
                            "silhouette": Amplitude(metric="silhouette"),
                            "heat": Amplitude(metric="heat"),
                            "perimage": Amplitude(metric="persistence_image"),
                            "perentropy": PersistenceEntropy(),
                            "points": NumberOfPoints(),
                            "coef": ComplexPolynomial(n_coefficients=polycoef, polynomial_type="S")}

    def mainf(self, prompt, pdf, update=False):
        tdf = pdf.replace(".perdiagram.pkf", ".topodescriptor.pkf")
        has_tdf = path.exists(tdf)

        if has_tdf and not update:
            print(prompt, "Been Processed")

            return None

        print(prompt, "Processing")

        with open(pdf, "rb") as rpkf:
            pdiagram = loads(decompress(rpkf.read()))

        if has_tdf:
            with open(tdf, "rb") as rpkf:
                tdescriptor = loads(decompress(rpkf.read()))
        else:
            tdescriptor = dict()

        ischanged = False
        for vky in pdiagram.keys():
            if all(ky not in vky for ky in ["original", "wavelet-HHH", "log-sigma-0-25"]):
                continue

            if tdescriptor.get(vky) is None:
                tdescriptor[vky] = dict()

            for sky in pdiagram[vky].keys():
                if tdescriptor[vky].get(sky) is None:
                    tdescriptor[vky][sky] = self.calculate_descriptor(pdiagram[vky][sky])

                    ischanged = True

        if ischanged:
            with open(tdf, "wb") as wpkf:
                wpkf.write(compress(dumps(tdescriptor)))

        print(prompt, "Done")

    def calculate_descriptor(self, subpdiagram):
        temp = {cky: self.calculators[cky].fit_transform(subpdiagram) for cky in self.calculators.keys()}

        tdkys = [ky for ky in temp.keys() if "coef" not in ky]
        subtdescriptor = {f"topodescriptor-{b}_{f}_{h}": float(temp[f][self.bindx[b], self.hindx[h]])
                          for b, f, h in product(self.bindx.keys(), tdkys, self.hindx.keys())}

        tdkys = [ky for ky in temp.keys() if "coef" in ky]
        subtdescriptor.update({f"topodescriptor-{b}_{f}_{c}": float(temp[f][self.bindx[b], self.cindx[c]])
                               for b, f, c in product(self.bindx.keys(), tdkys, self.cindx.keys())})

        return subtdescriptor


if __name__ == "__main__":
    srcs = [r"C:\Users\301PBDC\Desktop\dataset\images",
            r"C:\Users\301PBDC\Desktop\intraclass_correlation\images"]

    rst = r"I:\results"

    processor = ImageProcessor()
    topodescriptor = TopoDescriptor()

    error_check = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        for src in srcs:
            cdir = path.dirname(src)
            midpath = path.basename(cdir)
            tardir = f"{rst}\\{midpath}"

            pdfs = [f for folder in processor.return_dir(tardir)
                    for f in glob(f"{folder}\\*.perdiagram.pkf")]
            npdf = len(pdfs)

            for i, pf in enumerate(pdfs):
                prmt = f"{i + 1} in {npdf} ({(i + 1) / npdf:.2%}) {pf}"
                error_check.append([prmt, executor.submit(topodescriptor.mainf, prmt, pf, update=True)])

    for prmt, ck in error_check:
        try:
            ck.result()
        except Exception:
            print(prmt)
