import zlib
import glob
import pickle
from os import path

import pandas as pd

from imageprocessor import ImageProcessor

label = "CsqTexTop-project"

srcs = [r"C:\Users\301PBDC\Desktop\dataset\images",
        r"C:\Users\301PBDC\Desktop\intraclass_correlation\images"][1:]

rst = r"I:\results"

processor = ImageProcessor()

for src in srcs:
    cdir = path.dirname(src)
    midpath = path.basename(cdir)
    tdir = f"{rst}\\{midpath}"

    processor.check_dir(tdir)

    cdir = path.dirname(src)
    midpath = path.basename(cdir)
    tardir = f"{rst}\\{midpath}"

    if "intraclass_correlation" not in cdir:
        with open(f"{cdir}\\annotation_{label}.pkf", "rb") as rpkf:
            annotation = pickle.load(rpkf)

    for folder in processor.return_dir(tardir):
        for file_type in ["radiomics", "toporadiomics", "topomoment", "topodescriptor"]:
            fs = [f for f in glob.glob(f"{folder}\\*.{file_type}.pkf")]
            nf = len(fs)

            df_data = df_data_intra = df_data_supra = None

            for i, f in enumerate(fs):
                prmt = f"{i + 1} in {nf} ({(i + 1) / nf:.2%}) {f}"

                print(prmt, "Exporting", end="")

                lid = f.split("\\")[-1].split(".")[0]
                with open(f, "rb") as pkf:
                    data = pickle.loads(zlib.decompress(pkf.read()))

                vkys = list(data.keys())
                skys = {vky: list(data[vky].keys()) for vky in vkys}
                fkys = {vky: {sky: list(data[vky][sky].keys()) for sky in skys[vky]} for vky in vkys}

                if df_data is None:
                    df_data = {"set": [], "lid": [], "class": []}
                    df_data.update({ky: []
                                    for ky in [f"{vky}_{sky}_{fky}"
                                               for vky in fkys.keys()
                                               for sky in fkys[vky].keys()
                                               for fky in fkys[vky][sky]
                                               if sky != "supra"]})

                    df_data_intra = {"set": [], "lid": [], "class": []}
                    df_data_intra.update({ky: []
                                          for ky in [f"{vky}_{sky}_{fky}"
                                                     for vky in fkys.keys()
                                                     for sky in fkys[vky].keys()
                                                     for fky in fkys[vky][sky]
                                                     if sky == "intra"]})

                    df_data_supra = {"set": [], "lid": [], "class": []}
                    df_data_supra.update({ky: []
                                          for ky in [f"{vky}_{sky}_{fky}"
                                                     for vky in fkys.keys()
                                                     for sky in fkys[vky].keys()
                                                     for fky in fkys[vky][sky]
                                                     if sky == "supra"]})

                df_data["lid"].append(lid)
                df_data_intra["lid"].append(lid)
                df_data_supra["lid"].append(lid)

                if "intraclass_correlation" not in cdir:
                    df_data["set"].append(annotation[lid]["set"])
                    df_data_intra["set"].append(annotation[lid]["set"])
                    df_data_supra["set"].append(annotation[lid]["set"])

                    df_data["class"].append(annotation[lid]["class"])
                    df_data_intra["class"].append(annotation[lid]["class"])
                    df_data_supra["class"].append(annotation[lid]["class"])
                else:
                    df_data["set"].append(None)
                    df_data_intra["set"].append(None)
                    df_data_supra["set"].append(None)

                    df_data["class"].append(None)
                    df_data_intra["class"].append(None)
                    df_data_supra["class"].append(None)

                for vky in vkys:
                    for sky in skys[vky]:
                        for fky in fkys[vky][sky]:
                            if sky != "supra":
                                df_data[f"{vky}_{sky}_{fky}"].append(data[vky][sky][fky])
                                if sky == "intra":
                                    df_data_intra[f"{vky}_{sky}_{fky}"].append(data[vky][sky][fky])
                            else:
                                df_data_supra[f"{vky}_{sky}_{fky}"].append(data[vky][sky][fky])

                print(f"\r{prmt} Done     ")

            df = pd.DataFrame(df_data)
            df_intra = pd.DataFrame(df_data_intra)
            df_supra = pd.DataFrame(df_data_supra)

            with open(f"{folder}_{file_type}.pkf", "wb") as wpkf:
                pickle.dump(df, wpkf)

            with open(f"{folder}_{file_type}_intra.pkf", "wb") as wpkf:
                pickle.dump(df_intra, wpkf)

            with open(f"{folder}_{file_type}_supra.pkf", "wb") as wpkf:
                pickle.dump(df_supra, wpkf)
