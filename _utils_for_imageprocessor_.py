import SimpleITK as sitk
from classicalradiomics.imageoperations import getOriginalImage, getWaveletImage, getLoGImage



def _get_resample_size(orgn_siz, orgn_spc, rsmp_spc):
    return [round(osiz * ospc / rspc) for osiz, ospc, rspc in zip(orgn_siz, orgn_spc, rsmp_spc)]



