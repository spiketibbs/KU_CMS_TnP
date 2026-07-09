import awkward as ak
import numpy as np
from egamma_tnp.utils.vid_unpacked import vidUnpackedWP, veto_minus_iso_hoe, loose_minus_iso_hoe, tight_minus_iso_hoe


        
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

#def veto_minus_iso_hoe(ele_obj):
#    
#    vid_Unpacked = vidUnpackedWP(ele_obj) 
#    
#    #print(vidUnpacked.keys()) #testing before deleting the iso and hoe cut
#    
#    mask = (
#        (vid_Unpacked['MinPtCut'] >= 1) & 
#        (vid_Unpacked['GsfEleSCEtaMultiRangeCut'] >= 1) &
#        (vid_Unpacked['GsfEleDEtaInSeedCut'] >= 1) &
#        (vid_Unpacked['GsfEleDPhiInCut'] >= 1) &
#        (vid_Unpacked['GsfEleFull5x5SigmaIEtaIEtaCut'] >= 1) &
#        (vid_Unpacked['GsfEleEInverseMinusPInverseCut'] >= 1) &
#        (vid_Unpacked['GsfEleConversionVetoCut'] >= 1) &
#        (vid_Unpacked['GsfEleMissingHitsCut'] >= 1)
#    )
#
#    return mask
#
#def loose_minus_iso_hoe(ele_obj):
#
#    vid_Unpacked = vidUnpackedWP(ele_obj)
#
#    mask = (
#        (vid_Unpacked['MinPtCut'] == 2) &
#        (vid_Unpacked['GsfEleSCEtaMultiRangeCut'] >= 2) &
#        (vid_Unpacked['GsfEleDEtaInSeedCut'] >= 2) &
#        (vid_Unpacked['GsfEleDPhiInCut'] >= 2) &
#        (vid_Unpacked['GsfEleFull5x5SigmaIEtaIEtaCut'] >= 2) &
#        (vid_Unpacked['GsfEleEInverseMinusPInverseCut'] >= 2) &
#        (vid_Unpacked['GsfEleConversionVetoCut'] >= 2) &
#        (vid_Unpacked['GsfEleMissingHitsCut'] >= 2)
#    )
#
#    return mask
    
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

def lptelectron_masks(lpte_obj):
    masks = {}
    def lpte_sip3d(lpte_obj): return (np.sqrt((lpte_obj.dz/lpte_obj.dzErr)**2 + (lpte_obj.dxy/lpte_obj.dxyErr)**2))
    def lower_pt(lpte_obj): return  ((np.abs(lpte_obj.pt) < 5) & (np.abs(lpte_obj.pt) >= 2))
    def lowmed_pt(lpte_obj): return ((np.abs(lpte_obj.pt) < 7) & (np.abs(lpte_obj.pt) >= 5))
    def fareta(lpte_obj): return ((np.abs(lpte_obj.eta >= 1.48)) & (np.abs(lpte_obj.eta < 2.5)))
    def medeta(lpte_obj): return ((np.abs(lpte_obj.eta < 1.48)) & (np.abs(lpte_obj.eta >= 0.8)))
    def centeta(lpte_obj): return (np.abs(lpte_obj.eta < 0.8))
    masks["low_pt"] = ((np.abs(lpte_obj.pt) < 7) & (np.abs(lpte_obj.pt) >= 2))
    masks["lowest_pt"] = ((np.abs(lpte_obj.pt) < 5))
    masks["lower_pt"] = ((np.abs(lpte_obj.pt) >= 5))
    masks["eta_2p5"] = (np.abs(lpte_obj.eta)<2.5)
    masks["loose_sip3d"] = (lpte_sip3d(lpte_obj) < 6)
    masks["medium_sip3d"] = ((lpte_sip3d(lpte_obj) >= 2) & (lpte_sip3d(lpte_obj) < 6))
    masks["tight_sip3d"] = (lpte_sip3d(lpte_obj) < 2)
    masks["dxy_0p05"] = (np.abs(lpte_obj.dxy) < 0.05)
    masks["dz_0p1"] = (np.abs(lpte_obj.dz) < 0.1)
    masks["conv_veto"] = (lpte_obj.convVeto == 1 )
    masks["loose_ID"] = (lpte_obj.ID >= 1.5)
    def tight_ID_fareta_57(lpte_obj): return ((lpte_obj.ID > 3.5) & fareta(lpte_obj))
    def tight_ID_medeta_57(lpte_obj): return ((lpte_obj.ID > 3.0) & medeta(lpte_obj))
    def tight_ID_centeta_57(lpte_obj): return ((lpte_obj.ID > 2.3) & centeta(lpte_obj))
    masks["tight_ID_57"] = ((tight_ID_fareta_57(lpte_obj)) | (tight_ID_medeta_57(lpte_obj)) | (tight_ID_centeta_57(lpte_obj)))
    def tight_ID_medeta_25(lpte_obj): return ((lpte_obj.ID > 3.0) & medeta(lpte_obj))
    def tight_ID_loweta_25(lpte_obj): return ((lpte_obj.ID > 2.3) & centeta(lpte_obj))
    masks["tight_ID_loweta_25"] = ((tight_ID_medeta_25(lpte_obj)) | (tight_ID_loweta_25(lpte_obj)))
    masks["loose_miniPFRelIso"] =  (
        lpte_obj.miniPFRelIso_all * lpte_obj.pt) < (20 + (300/lpte_obj.pt)
                                                  )
    masks["tight_miniPFRelIso"] =  (lpte_obj.miniPFRelIso_all * lpte_obj.pt) <= 4
    return masks

def electron_masks(ele_obj):
        
    masks = {}

    masks["low_pt"] = ((np.abs(ele_obj.pt) < 20) & (np.abs(ele_obj.pt) >= 7))
    masks["high_pt"] = (np.abs(ele_obj.pt) >= 20)

    #Modified cutBased cuts
    masks["veto_no_iso_hoe"] =  veto_minus_iso_hoe(ele_obj)
    masks["loose_no_iso_hoe"] =  loose_minus_iso_hoe(ele_obj)
    masks["tight_no_iso_hoe"] =  tight_minus_iso_hoe(ele_obj)
    masks["wp90"] = (ele_obj.mvaIso_WP90)
    #masks["wp90"] = (ele_obj.mvaFall17V2Iso_WP90)
    

    #Primary vertex cuts 
    masks["losthit"] = (ele_obj.lostHits == 0)
    masks["convVeto"] = (ele_obj.convVeto == 1)
    masks["eta_2p5"] =  (np.abs(ele_obj.eta) < 2.5)
    #masks["loose_sip3d"] =  (ele_obj.sip3d < 8)
    masks["loose_sip3d"] =  (ele_obj.sip3d < 6)
    #masks["medium_sip3d"] =  ((ele_obj.sip3d >= 2) & (ele_obj.sip3d < 8))
    masks["medium_sip3d"] =  ((ele_obj.sip3d >= 2) & (ele_obj.sip3d < 6))
    masks["tight_sip3d"] =  (ele_obj.sip3d < 3)
    #masks["tight_sip3d"] =  (ele_obj.sip3d < 8)
    #masks["tight_sip3d"] =  (ele_obj.sip3d < 6)
    masks["dxy_0p05"] =  (np.abs(ele_obj.dxy) < 0.05)
    masks["dz_0p1"] =  (np.abs(ele_obj.dz) < 0.1)

    #Isolation cuts
    masks["loose_pfRelIso"] =  (
        ele_obj.pfRelIso03_all * ele_obj.pt) < (20 + (300/ele_obj.pt)
                                               )
    masks["tight_pfRelIso"] =  (ele_obj.pfRelIso03_all * ele_obj.pt) <= 4
    masks["loose_miniRelIso"] =  (
        ele_obj.miniPFRelIso_all * ele_obj.pt) < (20 + (300/ele_obj.pt)
                                               )
    masks["tight_miniPFRelIso"] =  (ele_obj.miniPFRelIso_all * ele_obj.pt) <= 4
    
    #masks[""] = 
    #masks[""] = 
    
    return masks
    
def lptelectron_categories(lpte_obj):

    masks = lptelectron_masks(lpte_obj)

    categories = {}

    categories["baselinep"] = (
        masks["low_pt"] &
        masks["eta_2p5"] &
        masks["loose_sip3d"] &
        masks["dxy_0p05"] &
        masks["dz_0p1"] &
        masks["conv_veto"] &
        masks["loose_ID"] &
        masks["loose_miniPFRelIso"]
    )
                              
    categories["silver"] = (
        categories["baselinep"] &
        masks["medium_sip3d"] &
        masks["tight_miniPFRelIso"] &
        (
            (masks["tight_ID_loweta_25"] & masks["lowest_pt"]) |
            (masks["tight_ID_57"] & masks["lower_pt"])
        )
    )

    categories["gold"] = (
        categories["baselinep"] &
        masks["tight_sip3d"] &
        masks["tight_miniPFRelIso"] &
        (
            (masks["tight_ID_loweta_25"] & masks["lowest_pt"]) |
            (masks["tight_ID_57"] & masks["lower_pt"])
        )
    )
                              
    categories["bronze"] = (
        categories["baselinep"] & ~(categories["gold"] | categories["silver"])
    )

    return categories
        
def electron_categories(ele_obj):

    masks = electron_masks(ele_obj)
    
    categories = {}

    categories["baselineps"] = ( 
        masks["losthit"] &
        masks["convVeto"] &
        #masks["veto_no_iso_hoe"] & 
        masks["loose_no_iso_hoe"] & 
        masks["loose_pfRelIso"] &
        masks["eta_2p5"] &
        masks["loose_sip3d"] &
        masks["dxy_0p05"] &
        masks["dz_0p1"]
    )
#    categories["baselinep"] = ( 
#        categories["baselineps"] &
#        masks["tight_sip3d"] &
#        #masks["tight_no_iso_hoe"] &
#        (
#        #    (masks["tight_no_iso_hoe"] & masks["tight_pfRelIso"] & masks["tight_miniPFRelIso"] & masks["low_pt"]) |
#            (masks["tight_no_iso_hoe"] & masks["low_pt"]) |
#            (masks["wp90"] & masks["high_pt"])
#        )
##        masks["tight_pfRelIso"] & 
##        masks["tight_miniPFRelIso"]
#    )
    categories["baselinep"] = ( 
        categories["baselineps"] &
        #masks["tight_sip3d"] &
        #(
            (masks["tight_no_iso_hoe"] & masks["tight_pfRelIso"] & masks["tight_miniPFRelIso"] & masks["low_pt"]) |
        #    (masks["tight_no_iso_hoe"] & masks["low_pt"]) |
            (masks["wp90"] & masks["high_pt"])
        #)
    )

    categories["silver"] = (
        categories["baselineps"] &
        masks["medium_sip3d"] &
        #masks["tight_no_iso_hoe"] &
        (
            (masks["tight_no_iso_hoe"] & masks["tight_pfRelIso"] & masks["tight_miniPFRelIso"] & masks["low_pt"]) |
            (masks["wp90"] & masks["high_pt"])
        )
#        masks["tight_pfRelIso"] &
#        masks["tight_miniPFRelIso"]
    )

    categories["gold"] = (
        categories["baselineps"] &
        masks["tight_sip3d"] &
        (
            (masks["tight_no_iso_hoe"] & masks["tight_pfRelIso"] & masks["tight_miniPFRelIso"] & masks["low_pt"]) |
        #    (masks["tight_no_iso_hoe"] & masks["low_pt"]) |
            (masks["wp90"] & masks["high_pt"])
        )
    )

    categories["golds"] = (
        categories["baselineps"] &
        masks["tight_sip3d"] &
        #masks["tight_no_iso_hoe"] &
        masks["wp90"]
        #masks["tight_no_iso_hoe"] & 
        #masks["tight_pfRelIso"] & 
        #masks["tight_miniPFRelIso"]
        #    (masks["tight_no_iso_hoe"] & masks["low_pt"]) |
        #(masks["wp90"] & masks["high_pt"])
        

    )

    categories["bronze"] = (
        categories["baselinep"] & ~(categories["gold"] | categories["silver"])
    )

    return categories


