# For converting the vidNestedWPBitmap value into 10 distinct cuts
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun3

import awkward as ak

def vidUnpackedWP(ele_obj):
    """
    Return a dictionary of the cuts in the electron cutBasedID,
    e.g. results["GsfEleEInverseMinusPInverseCut"] will be 0 (fail), 1, 2, 3, or 4 (tight)
    """
    results = {}
    for name, shift in zip(
        [
            "MinPtCut",
            "GsfEleSCEtaMultiRangeCut",
            "GsfEleDEtaInSeedCut",
            "GsfEleDPhiInCut",
            "GsfEleFull5x5SigmaIEtaIEtaCut",
            "GsfEleHadronicOverEMEnergyScaledCut",
            "GsfEleEInverseMinusPInverseCut",
            "GsfEleRelPFIsoScaledCut",
            "GsfEleConversionVetoCut",
            "GsfEleMissingHitsCut",
        ],
        range(0, 28, 3),
    ):
        results[name] = (ele_obj.vidNestedWPBitmap >> shift) & 0b111
    return results


def vidUnpackedWPSelection(electrons, level):
    """Return a dictionary of boolean masks for the electron cutBasedID,
    e.g. results["GsfEleEInverseMinusPInverseCut"] will be True if the result value is >= level
    """
    results = {}
    for name, cut_level in vidUnpackedWP(electrons).items():
        results[name] = cut_level >= level
        
    return results


#VETO ID without Isolation and without H/E cut:

def veto_minus_iso(ele_obj):
    
    vid_Unpacked = vidUnpackedWP(ele_obj)
    
    #print(vidUnpacked.keys()) #testing before deleting the iso and hoe cut
    
    mask = (
        (vid_Unpacked['MinPtCut'] >= 1) & 
        (vid_Unpacked['GsfEleSCEtaMultiRangeCut'] >= 1) &
        (vid_Unpacked['GsfEleDEtaInSeedCut'] >= 1) &
        (vid_Unpacked['GsfEleDPhiInCut'] >= 1) &
        (vid_Unpacked['GsfEleFull5x5SigmaIEtaIEtaCut'] >= 1) &
        (vid_Unpacked['GsfEleHadronicOverEMEnergyScaledCut'] >= 1) &
        (vid_Unpacked['GsfEleEInverseMinusPInverseCut'] >= 1) &
        (vid_Unpacked['GsfEleConversionVetoCut'] >= 1) &
        (vid_Unpacked['GsfEleMissingHitsCut'] >= 1)
    )

    return mask


def veto_minus_iso_hoe(ele_obj):
    
    vid_Unpacked = vidUnpackedWP(ele_obj) 
    
    #print(vidUnpacked.keys()) #testing before deleting the iso and hoe cut
    
    mask = (
        (vid_Unpacked['MinPtCut'] >= 1) & 
        (vid_Unpacked['GsfEleSCEtaMultiRangeCut'] >= 1) &
        (vid_Unpacked['GsfEleDEtaInSeedCut'] >= 1) &
        (vid_Unpacked['GsfEleDPhiInCut'] >= 1) &
        (vid_Unpacked['GsfEleFull5x5SigmaIEtaIEtaCut'] >= 1) &
        (vid_Unpacked['GsfEleEInverseMinusPInverseCut'] >= 1) &
        (vid_Unpacked['GsfEleConversionVetoCut'] >= 1) &
        (vid_Unpacked['GsfEleMissingHitsCut'] >= 1)
    )

    return mask


#TIGHT ID without Isolation and without H/E cut:

def tight_minus_iso(ele_obj):
    
    vid_Unpacked = vidUnpackedWP(ele_obj)
    
    #print(vidUnpacked.keys()) #testing before deleting the iso and hoe cut
    
    mask = (
        (vid_Unpacked['MinPtCut'] == 4) & 
        (vid_Unpacked['GsfEleSCEtaMultiRangeCut'] == 4) &
        (vid_Unpacked['GsfEleDEtaInSeedCut'] == 4) &
        (vid_Unpacked['GsfEleDPhiInCut'] == 4) &
        (vid_Unpacked['GsfEleFull5x5SigmaIEtaIEtaCut'] == 4) &
        (vid_Unpacked['GsfEleHadronicOverEMEnergyScaledCut'] == 4) &
        (vid_Unpacked['GsfEleEInverseMinusPInverseCut'] == 4) &
        (vid_Unpacked['GsfEleConversionVetoCut'] == 4) &
        (vid_Unpacked['GsfEleMissingHitsCut'] == 4)
    )

    return mask


def tight_minus_iso_hoe(ele_obj):
    
    vid_Unpacked = vidUnpackedWP(ele_obj)
    
    #print(vidUnpacked.keys()) #testing before deleting the iso and hoe cut
    
    mask = (
        (vid_Unpacked['MinPtCut'] == 4) & 
        (vid_Unpacked['GsfEleSCEtaMultiRangeCut'] == 4) &
        (vid_Unpacked['GsfEleDEtaInSeedCut'] == 4) &
        (vid_Unpacked['GsfEleDPhiInCut'] == 4) &
        (vid_Unpacked['GsfEleFull5x5SigmaIEtaIEtaCut'] == 4) &
        (vid_Unpacked['GsfEleEInverseMinusPInverseCut'] == 4) &
        (vid_Unpacked['GsfEleConversionVetoCut'] == 4) &
        (vid_Unpacked['GsfEleMissingHitsCut'] == 4)
    )

    return mask

