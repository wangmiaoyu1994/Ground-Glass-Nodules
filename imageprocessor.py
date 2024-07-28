import glob
from zlib import compress
from random import randint
from os import path, makedirs
from pickle import load, dumps
from concurrent.futures import ProcessPoolExecutor

import SimpleITK as sitk
from radiomics.imageoperations import getOriginalImage, getWaveletImage, getLoGImage


def _get_log_image(vol, seg):
    spacing = vol.GetSpacing()[0]
    return getLoGImage(vol, seg, sigma=[spacing * 0.5, spacing * 1.0, spacing * 2.0])


def _get_resample_size(orgn_siz, orgn_spc, rsmp_spc):
    return [round(osiz * ospc / rspc) for osiz, ospc, rspc in zip(orgn_siz, orgn_spc, rsmp_spc)]


def _vrsmp(vol, ref, **kwargs):
    return sitk.Resample(vol, ref, interpolator=sitk.sitkBSpline, **kwargs)


def _srsmp(seg, ref, **kwargs):
    return sitk.Resample(seg, ref, interpolator=sitk.sitkNearestNeighbor, **kwargs)


class ImageProcessor:
    def __init__(self, arbitrary=""):
        self.border = 6

        self.resampling_spacings = {"rsmp-uni-0.5": (0.5, 0.5, 0.5)}

        self.transformers = {"original": getOriginalImage, "wavelet": getWaveletImage, "log": _get_log_image}

        self.region_range = {"intra": None, "inter": (-2.0, 3.0), "extra": (0.01, 5.0), "supra": (-50.0, 5.0)}

        self.arbitrary = None
        self.randupper = None

        if path.exists(arbitrary):
            with open(arbitrary, "rb") as pkf:
                self.arbitrary = load(pkf)
                self.randupper = len(self.arbitrary)
                self.resampling_spacings["rsmp-arb-spc"] = None

    def check_dir(self, tardir):
        for ky in self.resampling_spacings.keys():
            if not path.exists(f"{tardir}\\{ky}"):
                makedirs(f"{tardir}\\{ky}")

    def return_dir(self, tardir):
        return [f"{tardir}\\{ky}" for ky in self.resampling_spacings.keys()]

    def resample(self, vol, seg, lung):
        label_map = sitk.BinaryImageToLabelMap(seg)

        crop_border = [round(self.border / s) for s in seg.GetSpacing()]
        crop_seg = sitk.LabelMapMask(label_map, seg, crop=True, cropBorder=crop_border)

        orgn_org = crop_seg.GetOrigin()
        orgn_siz = crop_seg.GetSize()
        orgn_spc = crop_seg.GetSpacing()
        orgn_drc = crop_seg.GetDirection()

        rsmp_siz = {ky: None if "raw" in ky else _get_resample_size(orgn_siz, orgn_spc, self.resampling_spacings[ky])
                    for ky in self.resampling_spacings.keys()}

        rsmp_para = {ky: {"outputOrigin": orgn_org,
                          "outputSpacing": self.resampling_spacings[ky],
                          "outputDirection": orgn_drc}
                     for ky in self.resampling_spacings.keys()}

        rsmp_vol = {ky: _vrsmp(vol, crop_seg) if "raw" in ky else _vrsmp(vol, rsmp_siz[ky], **rsmp_para[ky])
                    for ky in rsmp_para}

        rsmp_seg = {ky: crop_seg if "raw" in ky else _srsmp(seg, rsmp_siz[ky], **rsmp_para[ky])
                    for ky in rsmp_para}

        lung = sitk.Maximum(lung, seg)

        rsmp_lung = {ky: _srsmp(lung, crop_seg) if "raw" in ky else _srsmp(lung, rsmp_siz[ky], **rsmp_para[ky])
                     for ky in rsmp_para}

        return rsmp_vol, rsmp_seg, rsmp_lung

    def transform(self, vol, seg, lung):
        distn_seg = sitk.SignedMaurerDistanceMap(seg, squaredDistance=False, useImageSpacing=True)

        multiseg = {ky: seg if ky == "intra" else sitk.BinaryThreshold(distn_seg, *self.region_range[ky])
                    for ky in self.region_range.keys()}

        multiseg = {ky: seg if ky == "intra" else sitk.Minimum(multiseg[ky], lung)
                    for ky in multiseg.keys()}

        multivol = dict()

        for ky in self.transformers.keys():
            for trans_vol, trans_typ, _ in self.transformers[ky](vol, multiseg["supra"]):
                multivol[trans_typ] = sitk.Cast(trans_vol, pixelID=sitk.sitkInt16)

        return multivol, multiseg

    def resample_transform(self, volf, segbat, tardir):
        for prompt, segf in segbat:
            rdfn = path.basename(segf).replace(".seg.nrrd", ".radiodata.pkf")

            if not all(path.exists(f"{tardir}\\{ky}\\{rdfn}") for ky in self.resampling_spacings.keys()):
                break
        else:
            for prompt, segf in segbat:
                print(prompt, segf, "Been Processed")

            return None

        vol = sitk.ReadImage(volf)

        lungf = volf.replace(".nrrd", ".lung.seg.nrrd")
        lung = sitk.ReadImage(lungf)

        for prompt, segf in segbat:
            rdfn = path.basename(segf).replace(".seg.nrrd", ".radiodata.pkf")

            if all(path.exists(f"{tardir}\\{ky}\\{rdfn}") for ky in self.resampling_spacings.keys()):
                print(prompt, segf, "Been Processed")
                continue

            print(prompt, segf, "Processing")

            seg = sitk.ReadImage(segf)

            if self.arbitrary is not None:
                self.resampling_spacings["rsmp-arb-spc"] = self.arbitrary[randint(0, self.randupper)]

            rsmp_vol, rsmp_seg, rsmp_lung = self.resample(vol, seg, lung)

            for ky in rsmp_vol.keys():
                if path.exists(f"{tardir}\\{ky}\\{rdfn}"):
                    continue
                else:
                    multivol, multiseg = self.transform(rsmp_vol[ky], rsmp_seg[ky], rsmp_lung[ky])

                    with open(f"{tardir}\\{ky}\\{rdfn}", "wb") as pkf:
                        pkf.write(compress(dumps((multivol, multiseg))))

            print(prompt, segf, "Done")


if __name__ == "__main__":
    srcs = [r"C:\Users\301PBDC\Desktop\dataset\images",
            r"C:\Users\301PBDC\Desktop\intraclass_correlation\images"]

    rst = r"I:\results"

    processor = ImageProcessor()

    error_check = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        for src in srcs:
            cdir = path.dirname(src)
            midpath = path.basename(cdir)
            tdir = f"{rst}\\{midpath}"

            processor.check_dir(tdir)

            segfs = glob.glob(f"{src}\\*[0-9].seg.nrrd")
            nsegf = len(segfs)

            sbat = []

            suffix = segfs[0].split("\\")[-1].split("_")[-1]
            shared_volf = segfs[0].replace(f"_{suffix}", "_1.nrrd")

            for i, sf in enumerate(segfs):
                prmt = f"{i + 1} in {nsegf} ({(i + 1) / nsegf:.2%})"
                suffix = sf.split("\\")[-1].split("_")[-1]
                vf = sf.replace(f"_{suffix}", "_1.nrrd")

                if vf == shared_volf:
                    sbat.append((prmt, sf))
                else:
                    error_check.append([prmt, executor.submit(processor.resample_transform, shared_volf, sbat, tdir)])
                    shared_volf = vf
                    sbat = [(prmt, sf)]

            error_check.append([prmt, executor.submit(processor.resample_transform, shared_volf, sbat, tdir)])

    for prmt, ck in error_check:
        try:
            ck.result()
        except Exception:
            print(prmt)
