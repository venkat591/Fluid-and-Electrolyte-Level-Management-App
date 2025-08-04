# model.py

def predict_electrolyte_status(sodium, potassium, calcium, magnesium, chloride):
    result = {}

    # Sodium prediction (example ranges)
    if sodium < 135:
        result["Sodium"] = "Low"
    elif 135 <= sodium <= 145:
        result["Sodium"] = "Neutral"
    else:
        result["Sodium"] = "High"

    # Potassium prediction (example ranges)
    if potassium < 3.5:
        result["Potassium"] = "Low"
    elif 3.5 <= potassium <= 5.0:
        result["Potassium"] = "Neutral"
    else:
        result["Potassium"] = "High"

    # Calcium prediction (example ranges)
    if calcium < 8.5:
        result["Calcium"] = "Low"
    elif 8.5 <= calcium <= 10.5:
        result["Calcium"] = "Neutral"
    else:
        result["Calcium"] = "High"

    # Magnesium prediction (example ranges)
    if magnesium < 1.5:
        result["Magnesium"] = "Low"
    elif 1.5 <= magnesium <= 2.5:
        result["Magnesium"] = "Neutral"
    else:
        result["Magnesium"] = "High"

    # Chloride prediction (example ranges)
    if chloride < 98:
        result["Chloride"] = "Low"
    elif 98 <= chloride <= 106:
        result["Chloride"] = "Neutral"
    else:
        result["Chloride"] = "High"

    return result
