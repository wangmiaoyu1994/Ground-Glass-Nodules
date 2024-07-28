import traceback
from os import path
from pickle import loads, dumps
from itertools import product
from zlib import compress, decompress

import SimpleITK as sitk
from _utils_for_featureextractor import TopoMapper, Perputor
from _utils_for_featureextractor import RadiomicExtractor, QuantoExtractor
from _utils_for_featureextractor import ToporadiomicExtractor, TopoanalyticExtractor


class FeatureExtractor:
    def __init__(self):
        self.mapper = TopoMapper()
        self.perputor = Perputor()

        self.radiomic = RadiomicExtractor()
        self.quanto = QuantoExtractor()
        self.toporadiomic = ToporadiomicExtractor()
        self.topoanalytic = TopoanalyticExtractor()

    def extract(self, prmt, mvmsf):
        ff = mvmsf.replace(".multivol_multiseg.pkf", ".feature.pkf")
        hf = mvmsf.replace(".multivol_multiseg.pkf", ".homology.pkf")
        tf = mvmsf.replace(".multivol_multiseg.pkf", ".topomap.pkf")

        if path.exists(ff):
            print(prmt, mvmsf, "Been Processed")
            return None

        print(prmt, mvmsf, "Processing")

        try:
            with open(mvmsf, "rb") as pkf:
                mvol, mseg = loads(decompress(pkf.read()))

            del mseg["extra"], mseg["supra"]

            if path.exists(hf):
                with open(hf, "rb") as pkf:
                    mvry, msry, dgs = loads(decompress(pkf.read()))
            else:
                mvry = {vky: sitk.GetArrayFromImage(mvol[vky]) for vky in mvol.keys()}
                msry = {sky: sitk.GetArrayFromImage(mseg[sky]) for sky in mseg.keys()}
                dgs = {sky: {vky: None for vky in mvol.keys()} for sky in mseg.keys()}

            if path.exists(tf):
                with open(tf, "rb") as pkf:
                    tms = loads(decompress(pkf.read()))
            else:
                tms = {sky: {vky: None for vky in mvol.keys()} for sky in mseg.keys()}

            for vky, sky in product(mvol.keys(), mseg.keys()):
                if dgs[sky][vky] is None:
                    dgs[sky][vky] = self.perputor.compute_persistence(mvry[vky], msry[sky])

                if vky == "original" and tms[sky][vky] is None:
                    tms[sky][vky] = self.mapper.get_topomap_image(mvol[vky], mvry[vky], msry[sky])

            if not path.exists(hf):
                with open(hf, "wb") as pkf:
                    pkf.write(compress(dumps((mvry, msry, dgs))))

            if not path.exists(tf):
                with open(tf, "wb") as pkf:
                    pkf.write(compress(dumps(tms)))

            features = {sky: {vky: dict() for vky in mvol.keys()} for sky in mseg.keys()}

            for vky, sky in product(mvol.keys(), mseg.keys()):
                if vky == "original":
                    features[sky][vky].update(self.radiomic.extract(mvol[vky], mseg[sky], True))

                    axises = [features[sky][vky][f"radiomic_shape_{aky}AxisLength"]
                              for aky in ["Major", "Least", "Minor"]]
                    features[sky][vky].update(self.quanto.extract(mvry[vky], msry[sky], *axises))

                    features[sky][vky].update(self.toporadiomic.extract(tms[sky][vky], mseg[sky]))
                else:
                    features[sky][vky].update(self.radiomic.extract(mvol[vky], mseg[sky]))

                features[sky][vky].update(self.topoanalytic.extract(mvry[vky], msry[sky], dgs[sky][vky]))

            with open(ff, "wb") as pkf:
                pkf.write(compress(dumps(features)))

            print(prmt, mvmsf, "Done")
        except Exception:
            cdir = path.dirname(mvmsf)
            pdir = path.dirname(cdir)

            with open(f"{pdir}\\error_log.txt", "a") as ef:
                print(mvmsf, file=ef)
                traceback.print_exc(file=ef)
                print("\n", file=ef)

            print(prmt, mvmsf, "Error")
            return None
