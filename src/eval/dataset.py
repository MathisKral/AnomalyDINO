import os
import numpy as np


# FIXME: still bad practice...
def get_objects_from_dataset(dataset):
    if dataset == "MVTec":
        #objects = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
        objects = ["hazelnut"]
    elif dataset == "VisA":
        objects = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
    elif dataset == "Custom":
        objects = ["paper"]
    return objects

def get_dataset_info(dataset, preprocess, data_path=None):

    if preprocess not in ["informed", "agnostic", "masking_only", "informed_no_mask", "agnostic_no_mask", "force_no_mask_no_rotation", "force_mask_no_rotation", "force_no_mask_rotation", "force_mask_rotation"]:
        # masking only: deactivate rotation, apply masking like in informed/agnostic
        raise ValueError(f"Preprocessing '{preprocess}' not yet covered!")
    
    if dataset == "MVTec":
        objects = get_objects_from_dataset(dataset)
        object_anomalies = {"bottle": ["broken_large", "broken_small", "contamination"],
                            "cable": ["bent_wire", "cable_swap", "combined", "cut_inner_insulation", "cut_outer_insulation", "missing_wire", "missing_cable", "poke_insulation"],
                            "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
                            "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
                            "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
                            "hazelnut": ["crack", "cut", "hole", "print"],
                            "leather": ["color", "cut", "fold", "glue", "poke"],
                            "metal_nut": ["bent", "color", "flip", "scratch"],
                            "pill": ["color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
                            "screw": ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
                            "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
                            "toothbrush": ["defective"], 
                            "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
                            "wood": ["color", "combined", "hole", "liquid", "scratch"],
                            "zipper": ["broken_teeth", "combined", "fabric_border", "fabric_interior", "rough", "split_teeth", "squeezed_teeth"]
                            }

        if preprocess in ["agnostic", "informed", "masking_only"]:
            # Define Masking for the different objects -> determine with Masking Test (see Fig. 2 and discussion in the paper)
            # True: default masking (threshold the first PCA component > 10)
            # False: No masking will be applied
            masking_default = {"bottle": False,      
                                "cable": False,         # no masking
                                "capsule": True,        # default masking
                                "carpet": False,
                                "grid": False,
                                "hazelnut": True,
                                "leather": False,
                                "metal_nut": False,
                                "pill": True,
                                "screw": True,
                                "tile": False,
                                "toothbrush": True,
                                "transistor": False,
                                "wood": False,
                                "zipper": False
                                }
            
        if preprocess in ["informed", "informed_no_mask"]:
            rotation_default = {"bottle": False,
                                "cable": False, 
                                "capsule": False,
                                "carpet": False,
                                "grid": False,
                                "hazelnut": True,       # informed: hazelnut is rotated
                                "leather": False,
                                "metal_nut": False,
                                "pill": False,          # informed: all pills in train are oriented just the same
                                "screw": True,          # informed: screws in train are oriented differently
                                "tile": False,
                                "toothbrush": False,
                                "transistor": False,
                                "wood": False,
                                "zipper": False
                                }

        elif preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        elif preprocess == "masking_only":
            rotation_default = {o: False for o in objects}

        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}

    elif dataset == "VisA":
        objects = get_objects_from_dataset(dataset)
        object_anomalies = {"candle": ["bad"],
                            "capsules": ["bad"],
                            "cashew": ["bad"],
                            "chewinggum": ["bad"],
                            "fryum": ["bad"],
                            "macaroni1": ["bad"],
                            "macaroni2": ["bad"],
                            "pcb1": ["bad"],
                            "pcb2": ["bad"],
                            "pcb3": ["bad"],
                            "pcb4": ["bad"],
                            "pipe_fryum": ["bad"],
                            }

        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}
        else:
            masking_default = {o: True for o in objects}

        if preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        elif preprocess in ["informed", "masking_only", "informed_no_mask"]:
            rotation_default = {o: False for o in objects}

    elif dataset == "Custom":
        # CUSTOM DATASET ==========================================
        objects = get_objects_from_dataset(dataset)
        object_anomalies = {"paper": ["bad"]}
        masking_default = {o: False for o in objects} 
        rotation_default = {o: True for o in objects} 
        # =========================================================
    else:
        # raise ValueError(f"Dataset '{dataset}' not yet covered!")
        print(f"Dataset '{dataset}' not yet covered! Trying to infer objects from data_path, using default setting ('agnostic_no_mask').")
        preprocess = "agnostic_no_mask" # adapt this to your needs -> create custom settings as needed like for MVTec or VisA above

        if data_path is None:
            raise ValueError("Please provide 'data_path' when using custom dataset!")
        
        objects = []
        object_anomalies = {}
        masking_default = {}
        rotation_default = {}
        for o in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, o)):
                objects.append(o)
                masking_default[o] = False
                rotation_default[o] = True
                # infer anomaly types
                anomalies_path = os.path.join(data_path, o)
                anomaly_types = []
                for anomalytype in os.listdir(os.path.join(anomalies_path, "test")):
                    if anomalytype != "good" and os.path.isdir(os.path.join(anomalies_path, "test", anomalytype)):
                        anomaly_types.append(anomalytype)
                object_anomalies[o] = anomaly_types
        print(f"Found {len(objects)} objects: {objects} with on average {np.mean([len(object_anomalies[o]) for o in objects]):.1f} anomaly types per object.")

    if preprocess == "force_no_mask_no_rotation":
        masking_default = {o: False for o in objects}
        rotation_default = {o: False for o in objects}
    elif preprocess == "force_mask_no_rotation":
        masking_default = {o: True for o in objects}
        rotation_default = {o: False for o in objects}
    elif preprocess == "force_no_mask_rotation":
        masking_default = {o: False for o in objects}
        rotation_default = {o: True for o in objects}
    elif preprocess == "force_mask_rotation":
        masking_default = {o: True for o in objects}
        rotation_default = {o: True for o in objects}

    return objects, object_anomalies, masking_default, rotation_default