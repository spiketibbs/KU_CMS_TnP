import argparse
from pathlib import Path


def find_root_files(folder_name = None, type="MC", barrel="barrel_1"):

    base = Path("examples/nanoaod_filters_custom") / folder_name / type / "get_1d_pt_eta_phi_tnp_histograms_1/"
    print(base)
    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {base}")
    return sorted(base.glob(f"**/*{barrel}.root"))

def main():
    file_paths = [
    "DATA_barrel_1",                    "DATA_barrel_2",                    "DATA_endcap",                   
    "DATA_OLD_barrel_1",                "DATA_OLD_barrel_2",                "DATA_OLD_endcap",                
    "DATA_NEW_barrel_1",                "DATA_NEW_barrel_2",                "DATA_NEW_endcap",                  
    "DATA_NEW_2_barrel_1",              "DATA_NEW_2_barrel_2",              "DATA_NEW_2_endcap",                
    "MC_DY_barrel_1",                   "MC_DY_barrel_2",                   "MC_DY_endcap",                     
    "MC_DY2_2L_2J_barrel_1",            "MC_DY2_2L_2J_barrel_2",            "MC_DY2_2L_2J_endcap",           
    "MC_DY2_2L_4J_barrel_1",            "MC_DY2_2L_4J_barrel_2",            "MC_DY2_2L_4J_endcap",]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, choices=file_paths)
    parser.add_argument("--folder", required=True, default="blp_big")
    args = parser.parse_args()

    filepath = args.data
    parts = filepath.split("_")
    
    # Extract type (DATA or MC)
    type_parts = []
    for i, part in enumerate(parts):
        if part.startswith("barrel") or part == "endcap":
            barrel_index = i
            break
        type_parts.append(part)
    if "MC" in type_parts:
        type = "_".join(type_parts) + "_2023"
    elif "DATA" in type_parts:
        type = "_".join(type_parts) + "_2023D"
    
    # Extract barrel
    if parts[barrel_index] == "endcap":
        barrel = "endcap"
    else:
        barrel = f"barrel_{parts[barrel_index+1]}"
    
    # Extract n_d (either "gold_blp", "silver_blp", or default to "blp_big")
    folder_name = args.folder
    

    print(f"Type: {type}")
    print(f"Barrel: {barrel}")
    print(f"folder_name: {folder_name}")

    root_files = find_root_files(folder_name, type, barrel)

    if not root_files:
        print(f"No .root files found for {filepath}")
        return

    print(f"Found {len(root_files)} .root files:")
    for rf in root_files:
        print(f" - {rf}")

if __name__ == "__main__":
    main()