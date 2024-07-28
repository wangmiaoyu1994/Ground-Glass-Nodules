import zlib
import glob
import pickle

import numpy as np
import pandas as pd
import SimpleITK as sitk


def _pv_and_cr(vol_ary, seg_ary, threshod):
    pv = np.count_nonzero((vol_ary > threshod) * seg_ary)

    return pv, pv / np.count_nonzero(seg_ary)


temp_dict = {"lid": [], "n250_v": [], "n250_r": [], "n135_v": [], "n135_r": [],
             "size": [], "volume": [], "mass": []}
rdfs = glob.glob(r"I:\results\dataset\rsmp-uni-0.5\*.radiodata.pkf")
nrdf = len(rdfs)

for i, f in enumerate(rdfs):
    prmt = f"{i + 1} in {nrdf} ({(i + 1) / nrdf:.2%}) {f}"

    print(prmt, "Processing")

    temp_dict["lid"].append(f.split("\\")[-1].split(".")[0].upper())
    with open(f, "rb") as rpkf:
        mvms = pickle.loads(zlib.decompress(rpkf.read()))

    vry = sitk.GetArrayFromImage(mvms[0]["original"])
    sry = sitk.GetArrayFromImage(mvms[1]["intra"])

    n135_v, n135_p = _pv_and_cr(vry, sry, -135)
    n250_v, n250_p = _pv_and_cr(vry, sry, -250)

    temp_dict["n135_v"].append(n135_v * 0.125)
    temp_dict["n135_r"].append(n135_p)

    temp_dict["n250_v"].append(n250_v * 0.125)
    temp_dict["n250_r"].append(n250_p)

    with open(f.replace(".radiodata.pkf", ".radiomics.pkf"), "rb") as pkf:
        data = pickle.loads(zlib.decompress(pkf.read()))

    temp_dict["size"].append(data["original"]["intra"]["shape_MajorAxisLength"])
    temp_dict["volume"].append(data["original"]["intra"]["shape_MeshVolume"])

    mass = (data["original"]["intra"]["firstorder_Mean"] + 1000) / 1000 * data["original"]["intra"]["shape_MeshVolume"]
    temp_dict["mass"].append(mass)

    print(prmt, "Done")

df = pd.DataFrame(temp_dict)
df.to_excel(r"C:\Users\301PBDC\Desktop\dataset\ctr.xlsx", index=False)
