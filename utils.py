"""
utils.py - Backward Compatibility Shim
Provides legacy disease_dic mapping for older Flask routes.
"""

from src.disease_info import DISEASE_REGISTRY

# Map index to formatted description or name for legacy Flask template rendering
disease_dic = {}
for idx, (class_name, info) in enumerate(DISEASE_REGISTRY.items()):
    disease_dic[idx] = f"<h3>{info['crop']} - {info['disease']}</h3><p>{info['description']}</p>"
    disease_dic[class_name] = f"<h3>{info['crop']} - {info['disease']}</h3><p>{info['description']}</p>"
