"""
disease_info.py - Structured Agricultural Disease Intelligence Module
Contains evidence-based symptoms, prevention, treatment, and crop management guidelines
for PlantVillage crop disease classes.
"""

from typing import Dict, Any, List, Optional
import os
import pandas as pd

DISEASE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Apple___Apple_scab": {
        "crop": "Apple",
        "crop_scientific": "Malus domestica",
        "disease": "Apple Scab",
        "scientific_name": "Venturia inaequalis",
        "pathogen_type": "Fungal",
        "is_healthy": False,
        "severity": "medium",
        "emoji": "🍎",
        "description": "Apple scab is a serious fungal disease affecting Malus species worldwide, causing dark olive-green velvety lesions on leaves and scabby blemishes on fruit.",
        "symptoms": [
            "Olive-green to velvety dark brown lesions on upper leaf surfaces.",
            "Deformed, puckered, or curling leaves in young foliage.",
            "Premature yellowing and defoliation of severely infected leaves.",
            "Rough, corky brown scabs and fissures on developing fruit."
        ],
        "treatment": [
            {"title": "Sanitation & Pruning", "desc": "Rake and destroy fallen leaves in autumn to eliminate the primary overwintering inoculum of fungal ascospores."},
            {"title": "Protective Fungicide", "desc": "Apply approved protectant fungicides such as captan or copper formulations before wet spring infection events (bud break to petal fall)."},
            {"title": "Canopy Management", "desc": "Prune tree canopies annually to improve sunlight penetration and air movement, speeding up leaf drying."},
            {"title": "Consult Extension", "desc": "Follow local university agricultural extension spray-timing models (Mills periods) based on rainfall and temperature."}
        ],
        "prevention": [
            "Plant scab-resistant cultivars (e.g., Liberty, Prima, Freedom, Enterprise).",
            "Maintain wide tree spacing to maximize airflow.",
            "Avoid overhead irrigation; use drip or micro-sprinklers directed at soil.",
            "Apply autumn urea spray or shredded compost to accelerate leaf decomposition."
        ],
        "growth_care": {
            "temperature": "18°C – 24°C optimal for growth",
            "water": "Deep root soaking, 40–50 mm per 10 days",
            "soil_ph": "6.0 – 7.0 slightly acidic to neutral",
            "sunlight": "Full sun, 6 to 8 hours daily"
        }
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "crop_scientific": "Malus domestica",
        "disease": "Black Rot (Frogeye Leaf Spot)",
        "scientific_name": "Botryosphaeria obtusa",
        "pathogen_type": "Fungal",
        "is_healthy": False,
        "severity": "high",
        "emoji": "🍎",
        "description": "Black rot causes frogeye leaf spots, cankers on branches, and firm rotting of fruit often beginning at the calyx end.",
        "symptoms": [
            "Small purple specks on leaves expanding into circular tan lesions with distinct purple margins ('frogeye' spots).",
            "Firm brown rotted zones on fruit, eventually turning completely black with concentric rings.",
            "Sunken, reddish-brown cankers on branches and trunk bark."
        ],
        "treatment": [
            {"title": "Canker Removal", "desc": "Prune out diseased wood and dead branches at least 15 cm below visible cankers during dry weather; burn or discard infected wood."},
            {"title": "Mummy Disposal", "desc": "Collect and remove mummified fruits clinging to branches or fallen on the ground."},
            {"title": "Targeted Fungicides", "desc": "Use captan or thiophanate-methyl as recommended by local extension specialists from bloom until harvest."}
        ],
        "prevention": [
            "Prevent mechanical injury and sunscald to tree trunks.",
            "Maintain tree vigor through balanced soil nutrition and weed control.",
            "Regularly sterilize pruning shears with 70% isopropyl alcohol."
        ],
        "growth_care": {
            "temperature": "20°C – 28°C",
            "water": "Regular moderate irrigation; keep soil moist but well-drained",
            "soil_ph": "6.2 – 6.8",
            "sunlight": "Full sun"
        }
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "crop_scientific": "Malus domestica",
        "disease": "Cedar Apple Rust",
        "scientific_name": "Gymnosporangium juniperi-virginianae",
        "pathogen_type": "Fungal (Heteroecious Rust)",
        "is_healthy": False,
        "severity": "medium",
        "emoji": "🍎",
        "description": "A heteroecious rust requiring both apple trees and eastern red cedar/juniper trees to complete its complex two-year life cycle.",
        "symptoms": [
            "Bright orange-yellow circular spots on upper leaf surfaces appearing in late spring.",
            "Raised dark orange pustules forming within spots, oozing sticky sweet droplets.",
            "Tube-like fungal structures (aecia) protruding on the underside of affected leaves."
        ],
        "treatment": [
            {"title": "Targeted Fungicide", "desc": "Apply systemic myclobutanil or mancozeb at pink bud stage through early summer when cedar galls release orange gelatinous tendrils."},
            {"title": "Foliar Monitoring", "desc": "Inspect leaf undersides regularly and remove heavily infested leaves early in the season."}
        ],
        "prevention": [
            "Remove eastern red cedars within 1–2 km of orchards where practical.",
            "Select resistant apple varieties (e.g., Redfree, Liberty, William's Pride).",
            "Prune gelatinous rust galls from nearby juniper bushes in late winter."
        ],
        "growth_care": {
            "temperature": "15°C – 25°C",
            "water": "Consistent root watering",
            "soil_ph": "6.0 – 6.8",
            "sunlight": "Full sun"
        }
    },
    "Apple___healthy": {
        "crop": "Apple",
        "crop_scientific": "Malus domestica",
        "disease": "Healthy Leaf",
        "scientific_name": "N/A (Healthy Plant)",
        "pathogen_type": "None",
        "is_healthy": True,
        "severity": "none",
        "emoji": "🍏",
        "description": "Foliage exhibits vigorous chlorophyll pigmentation, uniform leaf margins, and absence of fungal, bacterial, or necrotic lesions.",
        "symptoms": [
            "Uniform rich green color across blade and veins.",
            "Intact cuticle without powdery coatings or spots.",
            "Sturdy petiole attachment and regular vascular structure."
        ],
        "treatment": [
            {"title": "Maintain Vigor", "desc": "Continue regular seasonal orchard nutrition, soil organic matter additions, and balanced irrigation."}
        ],
        "prevention": [
            "Monitor weekly for pest pressure (aphids, mites, codling moths).",
            "Maintain dormant winter sprays to protect upcoming spring foliage."
        ],
        "growth_care": {
            "temperature": "18°C – 26°C",
            "water": "25–35 mm water per week during fruit sizing",
            "soil_ph": "6.0 – 6.8",
            "sunlight": "6+ hours direct sunlight"
        }
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)",
        "crop_scientific": "Zea mays",
        "disease": "Common Rust",
        "scientific_name": "Puccinia sorghi",
        "pathogen_type": "Fungal (Rust)",
        "is_healthy": False,
        "severity": "high",
        "emoji": "🌽",
        "description": "Common rust produces airborne spores that travel long distances on air currents, causing widespread brick-red pustules on upper and lower leaf surfaces.",
        "symptoms": [
            "Oval to elongate cinnamon-brown to brick-red pustules (uredinia) scattered on both leaf surfaces.",
            "Ruptured leaf epidermal tissue with powdery orange-brown spore dust.",
            "Chlorosis surrounding pustules, turning brown and necrotic in severe cases."
        ],
        "treatment": [
            {"title": "Fungicide Threshold", "desc": "Apply triazole or strobilurin-based fungicides if pustules appear on lower leaves before tasseling and conditions remain humid."},
            {"title": "Crop Scout Routine", "desc": "Scout 20 plants across 5 field zones weekly during vegetative stages V6 through reproductive R2."}
        ],
        "prevention": [
            "Plant resistant corn hybrids with Rp gene resistance.",
            "Plant early in the season to mature past peak windborne spore migration.",
            "Maintain balanced nitrogen; excess nitrogen promotes succulent, rust-susceptible leaf tissue."
        ],
        "growth_care": {
            "temperature": "20°C – 30°C optimal",
            "water": "50 mm weekly during silking and grain fill",
            "soil_ph": "5.8 – 7.0",
            "sunlight": "Full direct sunlight (8+ hours)"
        }
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "crop_scientific": "Zea mays",
        "disease": "Northern Corn Leaf Blight",
        "scientific_name": "Exserohilum turcicum",
        "pathogen_type": "Fungal",
        "is_healthy": False,
        "severity": "high",
        "emoji": "🌽",
        "description": "A devastating fungal disease favoring moderate temperatures and humid conditions, producing large cigar-shaped necrotic lesions that reduce photosynthesis.",
        "symptoms": [
            "Long, elliptical, grayish-green or tan lesions (2.5 to 15 cm in length) with smooth margins.",
            "Dark, olive-colored fungal sporulation inside lesions under damp weather.",
            "Premature death of entire leaves when multiple lesions coalesce."
        ],
        "treatment": [
            {"title": "Foliar Fungicide", "desc": "Consider economic threshold spray (e.g., azoxystrobin + propiconazole) if lesions reach the ear leaf at or before silking."},
            {"title": "Crop Stubble Tillage", "desc": "Perform deep conservation tillage to bury crop debris where fungus overwinters."}
        ],
        "prevention": [
            "Select resistant hybrids bearing Ht1, Ht2, Ht3, or HtN resistance genes.",
            "Implement a minimum 1 to 2-year crop rotation out of corn (e.g., with soybeans).",
            "Avoid excessive plant densities that impede wind movement."
        ],
        "growth_care": {
            "temperature": "18°C – 27°C",
            "water": "Keep root zone hydrated; avoid evening overhead sprinklers",
            "soil_ph": "6.0 – 6.8",
            "sunlight": "Full sun"
        }
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)",
        "crop_scientific": "Zea mays",
        "disease": "Healthy Leaf",
        "scientific_name": "N/A (Healthy Plant)",
        "pathogen_type": "None",
        "is_healthy": True,
        "severity": "none",
        "emoji": "🌽",
        "description": "Healthy maize foliage featuring erect, dark green leaves with clean midribs and no lesions, blight, or nutrient deficiency signs.",
        "symptoms": [
            "Uniform dark green pigmentation.",
            "Smooth leaf surface without lesions, pustules, or chlorotic striping.",
            "Robust midrib structural integrity."
        ],
        "treatment": [
            {"title": "Fertilizer Schedule", "desc": "Maintain scheduled side-dress nitrogen applications based on pre-sidedress soil nitrate tests (PSNT)."}
        ],
        "prevention": [
            "Continue routine field scouting for fall armyworms and ear rot.",
            "Ensure adequate drainage to prevent waterlogging."
        ],
        "growth_care": {
            "temperature": "22°C – 32°C",
            "water": "Consistent irrigation, especially at tasseling",
            "soil_ph": "6.0 – 7.0",
            "sunlight": "Full sun"
        }
    },
    "Grape___Black_rot": {
        "crop": "Grape",
        "crop_scientific": "Vitis vinifera",
        "disease": "Grape Black Rot",
        "scientific_name": "Guignardia bidwellii",
        "pathogen_type": "Fungal",
        "is_healthy": False,
        "severity": "high",
        "emoji": "🍇",
        "description": "Black rot is one of the most destructive grape diseases in warm, humid climates, capable of destroying up to 80% of clusters if unchecked.",
        "symptoms": [
            "Small yellowish circular leaf spots turning reddish-brown with narrow dark margins.",
            "Tiny black specks (pycnidia) arranged in rings inside the leaf spots.",
            "Infected berry berries turn brown, then shrivel into hard, black, wrinkled 'mummies'."
        ],
        "treatment": [
            {"title": "Mummy Removal", "desc": "Prune out and destroy all mummified berries from vines and ground during winter dormancy."},
            {"title": "Early Season Protection", "desc": "Spray mancozeb, captan, or myclobutanil starting at 10 cm shoot growth through 4 weeks post-bloom."},
            {"title": "Canopy Opening", "desc": "Prune and tuck shoots into trellis wires to allow rapid leaf drying."}
        ],
        "prevention": [
            "Select vineyard sites with excellent solar exposure and air drainage.",
            "Maintain clean weed-free strip beneath vine rows.",
            "Avoid wild grapevines within 500 meters of the vineyard."
        ],
        "growth_care": {
            "temperature": "20°C – 30°C",
            "water": "Deep, infrequent irrigation (drip lines)",
            "soil_ph": "6.0 – 7.0",
            "sunlight": "Full sunlight"
        }
    },
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape",
        "crop_scientific": "Vitis vinifera",
        "disease": "Esca (Black Measles)",
        "scientific_name": "Phaeomoniella chlamydospora & Phaeoacremonium minimum",
        "pathogen_type": "Fungal Complex",
        "is_healthy": False,
        "severity": "high",
        "emoji": "🍇",
        "description": "A complex trunk disease causing 'tiger-stripe' interveinal chlorosis/necrosis on leaves and dark specks on berries, often leading to vine apoplexy (sudden wilt).",
        "symptoms": [
            "'Tiger-striped' leaves with interveinal yellow or red patches bordered by necrotic brown margins.",
            "Dark brown or purple spots ('measles') sprinkled across berry skins.",
            "Sudden wilting and collapse of entire shoots or vines during hot midsummer spells."
        ],
        "treatment": [
            {"title": "Wound Protection", "desc": "Paint pruning wounds with antifungal sealants or Trichoderma formulations immediately after cutting."},
            {"title": "Remedial Surgery", "desc": "Trunk renewal: cut back diseased wood until reaching healthy white tissue and train a new sucker."},
            {"title": "Marking Infected Vines", "desc": "Tag infected vines in summer and prune them last during winter to avoid spreading spores."}
        ],
        "prevention": [
            "Prune late in the dormant season (February/March) when wounds heal faster.",
            "Avoid pruning vines during wet rain events.",
            "Purchase certified disease-free nursery rootstocks."
        ],
        "growth_care": {
            "temperature": "20°C – 32°C",
            "water": "Controlled deficit irrigation post fruit-set",
            "soil_ph": "6.2 – 7.2",
            "sunlight": "Full sun"
        }
    },
    "Grape___healthy": {
        "crop": "Grape",
        "crop_scientific": "Vitis vinifera",
        "disease": "Healthy Leaf",
        "scientific_name": "N/A (Healthy Plant)",
        "pathogen_type": "None",
        "is_healthy": True,
        "severity": "none",
        "emoji": "🍇",
        "description": "Healthy grapevine foliage displaying vibrant palmate green leaves without chlorosis, powdery efflorescence, or lesions.",
        "symptoms": [
            "Uniform leaf greening and supple leaf texture.",
            "Absence of interveinal discoloration or curling margins.",
            "Vigorous shoot growth and clean tendrils."
        ],
        "treatment": [
            {"title": "Maintenance Care", "desc": "Continue trellis canopy management, leaf pulling around fruiting zones, and pest scouting."}
        ],
        "prevention": [
            "Monitor for early signs of downy mildew or powdery mildew.",
            "Ensure proper vine balance between crop load and leaf area."
        ],
        "growth_care": {
            "temperature": "20°C – 28°C",
            "water": "Targeted drip irrigation based on evapotranspiration (ETc)",
            "soil_ph": "6.0 – 7.0",
            "sunlight": "7–9 hours sun"
        }
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "crop_scientific": "Solanum tuberosum",
        "disease": "Early Blight",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "Fungal",
        "is_healthy": False,
        "severity": "medium",
        "emoji": "🥔",
        "description": "Early blight affects potato foliage and tubers, causing characteristic dark target-patterned concentric rings primarily starting on older lower leaves.",
        "symptoms": [
            "Circular brown or black spots with characteristic concentric rings ('target-board' appearance).",
            "Yellow halo encircling individual leaf lesions.",
            "Senescence and premature drying of lower foliage spreading upward.",
            "Dry, dark, leathery sunken lesions on harvested tubers."
        ],
        "treatment": [
            {"title": "Protectant Sprays", "desc": "Apply protectant fungicides like chlorothalonil or mancozeb at row closure or when first spots appear on lower leaves."},
            {"title": "Defoliation & Harvest Gap", "desc": "Kill vines 2–3 weeks before harvest to allow tuber skins to set and prevent spore transfer onto tubers."}
        ],
        "prevention": [
            "Plant certified disease-free seed tubers.",
            "Practice minimum 3-year crop rotation out of solanaceous crops.",
            "Avoid water stress; keep plants adequately fertilized with balanced nitrogen and potassium."
        ],
        "growth_care": {
            "temperature": "18°C – 24°C",
            "water": "25–40 mm per week; maintain uniform soil moisture",
            "soil_ph": "5.2 – 6.5",
            "sunlight": "Full sun"
        }
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "crop_scientific": "Solanum tuberosum",
        "disease": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "Oomycete",
        "is_healthy": False,
        "severity": "high",
        "emoji": "🥔",
        "description": "Historically infamous as the pathogen behind the Irish Potato Famine, late blight is an aggressive oomycete that can devastate an entire field in days under cool, wet conditions.",
        "symptoms": [
            "Water-soaked irregular dark green to purplish-black lesions on leaves and stems.",
            "Delicate white fungal-like sporulation on the undersides of leaves during high humidity.",
            "Rapid collapse and browning of foliar canopy with a distinct decaying odor.",
            "Granular brown rot penetrating into tuber flesh beneath the skin."
        ],
        "treatment": [
            {"title": "Immediate Fungicide Action", "desc": "Apply systemic oomycete-targeted compounds (e.g., cymoxanil, mandipropamid, or metalaxyl) immediately upon local blight advisory warning."},
            {"title": "Vine Destruction", "desc": "If more than 5% of canopy is infected, destroy vines chemically or mechanically to protect tubers below ground."},
            {"title": "Do Not Compost", "desc": "Burn, deeply bury, or dispose of infected plant matter; never leave in open compost."}
        ],
        "prevention": [
            "Plant late blight-resistant potato varieties (e.g., Defender, Sarpo Mira, Elba).",
            "Eliminate cull piles and volunteer potatoes in surrounding fields.",
            "Use weather-based blight forecast warning systems (e.g., Blitecast)."
        ],
        "growth_care": {
            "temperature": "15°C – 22°C (disease thrives in 15–20°C wet conditions)",
            "water": "Drip irrigate early in the morning so canopy dries quickly",
            "soil_ph": "5.5 – 6.5",
            "sunlight": "6+ hours sunlight"
        }
    },
    "Potato___healthy": {
        "crop": "Potato",
        "crop_scientific": "Solanum tuberosum",
        "disease": "Healthy Leaf",
        "scientific_name": "N/A (Healthy Plant)",
        "pathogen_type": "None",
        "is_healthy": True,
        "severity": "none",
        "emoji": "🥔",
        "description": "Vigorous potato foliage with dense green compound leaves, sturdy stems, and no visible spotting or blight symptoms.",
        "symptoms": [
            "Vibrant green compound leaves with clean margins.",
            "Turgid erect stems with no vascular discolouration.",
            "Healthy stolon and tuber initiation underground."
        ],
        "treatment": [
            {"title": "Hilling & Nutrition", "desc": "Hill soil around stems to protect developing tubers from sunlight (greening) and fungal spores."}
        ],
        "prevention": [
            "Scout fields twice weekly during cool, misty weather.",
            "Maintain soil moisture evenly during tuber bulking stage."
        ],
        "growth_care": {
            "temperature": "16°C – 22°C",
            "water": "Regular moderate irrigation, avoiding waterlogging",
            "soil_ph": "5.5 – 6.5",
            "sunlight": "Full sun"
        }
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "crop_scientific": "Solanum lycopersicum",
        "disease": "Bacterial Spot",
        "scientific_name": "Xanthomonas perforans / euvesicatoria",
        "pathogen_type": "Bacterial",
        "is_healthy": False,
        "severity": "high",
        "emoji": "🍅",
        "description": "Bacterial spot is a widespread disease in warm, rainy climates, causing water-soaked leaf spots, defoliation, and unmarketable scabby fruit lesions.",
        "symptoms": [
            "Small (less than 3 mm), dark, greasy, water-soaked spots on leaves that turn brown or black.",
            "Lesions develop a subtle yellow halo; centers may drop out creating a shot-hole effect.",
            "Severe blighting, leaf curling, and sunscald on unprotected fruit.",
            "Small raised black blisters on green fruit expanding into brown scabby spots."
        ],
        "treatment": [
            {"title": "Copper-Bactericide Spray", "desc": "Apply fixed copper tank-mixed with mancozeb, which increases soluble copper activity against bacterial strains."},
            {"title": "Bacteriophage Application", "desc": "Use registered agricultural bacteriophages (e.g., AgriPhage) targeted specifically against Xanthomonas strains."}
        ],
        "prevention": [
            "Use certified pathogen-free seeds treated with hot water (50°C for 25 minutes).",
            "Never work in or prune tomato fields when plants are wet with dew or rain.",
            "Rotate beds away from solanaceous crops for 2 to 3 years."
        ],
        "growth_care": {
            "temperature": "24°C – 32°C",
            "water": "Irrigate at ground level; do not splash soil onto lower leaves",
            "soil_ph": "6.0 – 6.8",
            "sunlight": "Full sun (8 hours)"
        }
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "crop_scientific": "Solanum lycopersicum",
        "disease": "Early Blight",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "Fungal",
        "is_healthy": False,
        "severity": "medium",
        "emoji": "🍅",
        "description": "A very common fungal disease affecting tomatoes, typically starting on older foliage near the ground and displaying characteristic concentric target rings.",
        "symptoms": [
            "Circular brown to dark brown spots with concentric rings ('target pattern') on older lower leaves.",
            "Surrounding leaf tissue turns yellow; affected leaves prematurely drop off.",
            "Dark, sunken, leathery cankers on stems near the soil line in seedlings ('collar rot').",
            "Dark sunken leathery lesions at the stem attachment point on ripe or green fruit."
        ],
        "treatment": [
            {"title": "Lower Leaf Pruning", "desc": "Prune off the bottom 20–30 cm of foliage to eliminate soil-contact splash and increase air circulation."},
            {"title": "Bio-fungicides & Copper", "desc": "Apply Bacillus subtilis or copper hydroxide protectant sprays at 7–10 day intervals during humid weather."},
            {"title": "Mulching", "desc": "Lay a 5–7 cm thick layer of clean straw, wood chips, or plastic mulch to prevent spores in the soil from splashing up."}
        ],
        "prevention": [
            "Space tomato plants 60–90 cm apart and stake or cage them off the ground.",
            "Avoid overhead sprinkler irrigation; water with soaker hoses or drip tape.",
            "Practice a 3-year crop rotation."
        ],
        "growth_care": {
            "temperature": "20°C – 28°C",
            "water": "2.5 cm water per week deeply at the base",
            "soil_ph": "6.2 – 6.8",
            "sunlight": "Full sun"
        }
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "crop_scientific": "Solanum lycopersicum",
        "disease": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "Oomycete",
        "is_healthy": False,
        "severity": "high",
        "emoji": "🍅",
        "description": "Late blight is a fast-moving, destructive oomycete disease capable of killing mature tomato plants and rotting entire fruit trusses within one to two weeks during wet conditions.",
        "symptoms": [
            "Large, irregularly shaped, water-soaked greenish-black lesions on leaves and petioles.",
            "Delicate white fungal-like growth visible on the undersides of infected leaves during morning or high humidity.",
            "Dark brown greasy-looking lesions on stems and branches.",
            "Golden-brown to dark greasy rotting patches on green and ripe tomato fruits."
        ],
        "treatment": [
            {"title": "Immediate Quarantine & Pruning", "desc": "Bag and cut infected leaves or entire plants in sealed bags immediately. Do not compost."},
            {"title": "Oomycete Fungicides", "desc": "Apply systemic fungicides containing cyazofamid, fluopicolide, or chlorothalonil immediately following local blight alert."},
            {"title": "Airflow Promotion", "desc": "Prune suckers and stake plants to maximize drying speed after rain or morning dew."}
        ],
        "prevention": [
            "Grow resistant tomato cultivars (e.g., Defiant PhR, Mountain Magic, Iron Lady, Plum Regal).",
            "Never plant tomatoes adjacent to potato patches.",
            "Water only at soil level in early morning hours."
        ],
        "growth_care": {
            "temperature": "18°C – 25°C",
            "water": "Consistent drip irrigation to maintain even soil moisture",
            "soil_ph": "6.0 – 6.8",
            "sunlight": "Full sun"
        }
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "crop_scientific": "Solanum lycopersicum",
        "disease": "Healthy Leaf",
        "scientific_name": "N/A (Healthy Plant)",
        "pathogen_type": "None",
        "is_healthy": True,
        "severity": "none",
        "emoji": "🍅",
        "description": "Healthy tomato foliage with deep green leaflets, strong aromatic scent, active vegetative growth, and no signs of bacterial specks, blights, or viruses.",
        "symptoms": [
            "Uniform vibrant green leaves with no spotting, mottling, or curled margins.",
            "Firm, upright main stem and vigorous lateral shoot development.",
            "Healthy yellow flower clusters and firm green fruit set."
        ],
        "treatment": [
            {"title": "Nutritional Balance", "desc": "Apply balanced fertilizer (e.g. NPK 5-10-10 or organic compost) with adequate calcium to prevent blossom end rot."}
        ],
        "prevention": [
            "Stake or cage plants off the soil.",
            "Maintain organic mulch layer around base.",
            "Scout leaf undersides weekly for hornworms, whiteflies, or aphids."
        ],
        "growth_care": {
            "temperature": "21°C – 29°C",
            "water": "25–35 mm per week, consistently to avoid fruit cracking",
            "soil_ph": "6.2 – 6.8",
            "sunlight": "Full sun, 8+ hours per day"
        }
    }
}

DEFAULT_UNKNOWN_DISEASE: Dict[str, Any] = {
    "crop": "Unknown Plant",
    "crop_scientific": "Plantae",
    "disease": "Unidentified / Low Confidence Condition",
    "scientific_name": "Undetermined",
    "pathogen_type": "Unknown",
    "is_healthy": False,
    "severity": "medium",
    "emoji": "🌿",
    "description": "The uploaded leaf image does not clearly match any of our supported crop disease classes with sufficient statistical confidence.",
    "symptoms": [
        "Visual symptoms do not match known class patterns with high probability.",
        "Image might contain background clutter, non-leaf objects, or an unsupported plant species."
    ],
    "treatment": [
        {"title": "Verify Leaf Image", "desc": "Take a clear, close-up photograph of a single leaf against a plain background in bright, indirect natural light."},
        {"title": "Consult Extension Expert", "desc": "Bring a physical sample in a sealed plastic bag to your local agricultural extension service (e.g. KVK, university agriculture department)."}
    ],
    "prevention": [
        "Upload images only for supported crops: Apple, Corn, Grape, Potato, Tomato.",
        "Ensure the camera is focused directly on the infected region of the leaf."
    ],
    "growth_care": {
        "temperature": "N/A",
        "water": "N/A",
        "soil_ph": "N/A",
        "sunlight": "N/A"
    }
}


def get_all_supported_classes() -> List[str]:
    """Return sorted list of supported model class IDs."""
    return sorted(list(DISEASE_REGISTRY.keys()))


def get_disease_info(class_name: str) -> Dict[str, Any]:
    """Retrieve structured disease intelligence dictionary for a given class ID."""
    if class_name in DISEASE_REGISTRY:
        return DISEASE_REGISTRY[class_name]
    
    # Try normalized / fuzzy matching
    clean_key = class_name.strip()
    for key, data in DISEASE_REGISTRY.items():
        if clean_key.lower() == key.lower():
            return data
        if clean_key.lower() in key.lower() or key.lower() in clean_key.lower():
            return data

    return DEFAULT_UNKNOWN_DISEASE


def export_csv_compatibility_files(base_dir: str = ".") -> None:
    """
    Generate disease_info.csv and supplement_info.csv in base_dir
    to ensure full backward compatibility with any legacy scripts.
    """
    disease_rows = []
    supplement_rows = []

    for idx, (class_name, info) in enumerate(DISEASE_REGISTRY.items()):
        disease_rows.append({
            "index": idx,
            "class_name": class_name,
            "crop": info["crop"],
            "disease_name": f"{info['crop']} - {info['disease']}",
            "description": info["description"],
            "pathogen": info["pathogen_type"],
            "severity": info["severity"],
            "is_healthy": info["is_healthy"]
        })
        
        # Primary treatment suggestion as supplement recommendation
        treat_title = info["treatment"][0]["title"] if info["treatment"] else "General Care"
        treat_desc = info["treatment"][0]["desc"] if info["treatment"] else "Maintain plant health"
        supplement_rows.append({
            "index": idx,
            "disease_name": f"{info['crop']} - {info['disease']}",
            "supplement name": f"{info['crop']} Protective Care: {treat_title}",
            "supplement image": "https://images.unsplash.com/photo-1585336261026-6819b16ea984?auto=format&fit=crop&w=400&q=80",
            "buy link": "https://www.agri-care-sample.org/resources",
            "notes": treat_desc
        })

    disease_df = pd.DataFrame(disease_rows)
    supplement_df = pd.DataFrame(supplement_rows)

    d_path = os.path.join(base_dir, "disease_info.csv")
    s_path = os.path.join(base_dir, "supplement_info.csv")

    disease_df.to_csv(d_path, index=False, encoding="utf-8")
    supplement_df.to_csv(s_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    export_csv_compatibility_files(".")
    print(f"Exported compatibility CSVs for {len(DISEASE_REGISTRY)} classes successfully.")
