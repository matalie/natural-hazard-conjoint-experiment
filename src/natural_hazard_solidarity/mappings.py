"""Survey mappings from legacy helpers/map_data.py; no import-time file IO."""
import numpy as np

VALID_COLUMNS = [
    "sensitivity_nh_1", "sensitivity_nh_2", "sensitivity_nh_3", "sensitivity_nh_4",
    "costs_cc_policy_1", "finan_vulnerability_1",
    "lreco_1", "lreco_2",
    "climatechange_nh_1", "climatechange_nh_2", "climatechange_nh_3", 
    "psycho_distance_1", "psycho_distance_2", "psycho_distance_3", "psycho_distance_4",
    "gal_tan_1", "gal_tan_2", "gal_tan_3", "gal_tan_4",
    "deservingness_3", "deservingness_4",
    "deservingness_5", "deservingness_6",
    "deservingness_7", "deservingness_8",
    "deservingness_9", "deservingness_10",
    "deservingness_11", "deservingness_12",        
]

NUM_COLUMNS = [
    "number_household_1_TEXT"
]

LIKERT_MAP = {
    "1 - Extremely unlikely": 1,
    "1 - No trust at all": 1,
    "1 - Strongly disagree": 1,
    "1 - Totally unacceptable": 1,
    "1 - Very unlikely": 1,
    "1 - Very difficult": 1,
    "1 - Not worried at all": 1,
    "1 - Lowest Priority":1,
    "6 - Totally acceptable": 6,
    "6 - Strongly agree": 6,
    "6 - Very likely": 6,
    "6 - Very easy": 6,
    "6 - Extremely worried": 6,
    "6 - Extremely likely": 6,
    "6 - Highest Priority":6,
    "6 - Complete trust":6,
    "I don't know": np.nan,
    "Prefer not to say": np.nan,
    "I don't now": np.nan,
    "7": np.nan
}

NH_EXPERIENCE_MAP = {
    'No': 1,
    'Yes, due to a different natural hazard such as': 0,
    'Yes, due to flooding,Yes, due to a different natural hazard such as': 0,
    'Yes, due to flooding': 0,
    'Yes, due to flooding,Yes, due to a landslide,Yes, due to a different natural hazard such as': 0,
    'Yes, due to a debris flow' 'Yes, due to a landslide': 0,
    'Yes, due to flooding,Yes, due to a debris flow,Yes, due to a landslide': 0,
    'Yes, due to flooding,Yes, due to a landslide': 0,
    'Yes, due to flooding,Yes, due to a debris flow': 0,
    'Yes, due to flooding,Yes, due to a debris flow,Yes, due to a landslide,Yes, due to a different natural hazard such as': 0,
}

ARTICLES_MAP = {
    '1 - 5 items': 1, 
    '6 - 10 items': 2, 
    '11 - 15 items': 3,
    '16 - 20 items': 4,
    '21 - 25 items': 5, 
    '26 - 30 items': 6,
    'More than 30 items': 6, 
    np.nan: np.nan    
}

CONVERSATIONS_MAP = {
    '1 - 3 conversations': 1, 
    '4 - 6 conversations': 2,
    '7 - 9 conversations': 3, 
    '10 - 12 conversations': 4, 
    '12 - 15 conversations': 5, 
    '16 - 18 conversations': 6, 
    '19 - 21 conversations': 6,
    'More than 22 conversations': 6,
    np.nan: np.nan
}

DEMOGRAPHICS_DICT = {
    # gender
    "Female": "Female",
    "Male": "Male",
    "Non-binary / Other": np.nan,
    "Prefer not to say": np.nan,  # beachte Leerzeichen wie im R-Code
    # age
    "18 - 34": "18 - 34",
    "35 - 49": "35 - 49",
    "50 or older": "50+",
    # education
    "No high school diploma": "Below Secondary",
    "Vocational training or apprenticeship": "Vocational training or apprenticeship",
    "High school diploma": "High school diploma",
    "Bachelor's degree": "University degree",
    "Master's degree": "University degree",
    "Doctoral or professional degree (e.g. PhD, MD, JD)": "University degree",
    "Prefer not to say": np.nan,
    # income
    # TODO: what source?
    # "Less than CHF 50,000": 45000,
    # "CHF 50,000 - CHF 70,000": 60000,
    # "CHF 70,000 - CHF 100,000": 85000,
    # "CHF 100,001 - CHF 150,000": 125000,
    # "CHF 150,001 - CHF 250,000": 200000,
    # "More than CHF 250,000": 300000,
    "Less than CHF 50,000": "Low",  #"Low",
    "CHF 50,000 - CHF 70,000": "Low", # "Low",
    "CHF 70,000 - CHF 100,000": "Mid", #"Mid",
    "CHF 100,001 - CHF 150,000": "Mid", #"Mid",
    "CHF 150,001 - CHF 250,000": "High", #"High", 
    "More than CHF 250,000": "High", #"High",
    # "Less than CHF 50,000": 3,  #"Low",
    # "CHF 50,000 - CHF 70,000": 3, # "Low",
    # "CHF 70,000 - CHF 100,000": 2, #"Mid",
    # "CHF 100,001 - CHF 150,000": 2, #"Mid",
    # "CHF 150,001 - CHF 250,000": 1, #"High", 
    # "More than CHF 250,000": 1, #"High",
    "I don't know": np.nan,
    "Prefer not to say": np.nan,
    # language
    "Deutsch": "German",
    "Français": "French",
    "English": "English",
    "Italiano": "Italian",
    # region
    "German-speaking region": "German-speaking region",
    "Italian-speaking region": "Italian-speaking region",
    "French-speaking region": "French-speaking region",
    "Romansh-speaking region": "Romansh-speaking region",
    # political
    "Social Democratic Party (SP)": "Left",
    "The Greens (GPS)": "Left",
    "Green Liberals (GLP)": "Liberal",
    "The Liberals (FDP)": "Liberal",
    "The Middle Party (merger between the Christian Democratic People's Party (CVP) and the Civic Democratic Party (BDP))": "Liberal",
    "Swiss People's Party (SVP)": "Conservative",
    "Federal Democratic Union (EDU)": "Conservative",
    "Evangelical People's Party of Switzerland (EPP)": "Conservative",
    "Mouvement Citoyens Genevois (MCG)": "Conservative",
    "Lega dei Ticinesi (Lega)": "Conservative",
    "Others, such as": np.nan,
    "I don't feel close to any party": np.nan,
    "Prefer not to say": np.nan,
}

TRANSLATION_DICT = {
    # Costs
    "Tous les citoyens paient le même montant": "All people pay the same amount",
    "Alle Menschen zahlen den gleichen Betrag": "All people pay the same amount",
    "Tutte le persone pagano lo stesso importo": "All people pay the same amount",

    "Les personnes paient proportionnellement à leurs revenus": "People pay proportionally to their income",
    "Menschen zahlen proportional zu ihrem Einkommen": "People pay proportionally to their income",
    "Le persone pagano in proporzione al loro reddito": "People pay proportionally to their income",

    "Les personnes et entreprises bénéficiant des mesures de protection": "People & companies being protected by protective measures",
    "Menschen und Unternehmen, die von Schutzmaßnahmen profitieren": "People & companies being protected by protective measures",
    "Le persone e aziende che beneficiano di misure di protezione": "People & companies being protected by protective measures",

    "Les personnes paient proportionnellement à leurs émissions de CO2": "People pay proportionally to their CO2 emissions",
    "Menschen zahlen proportional zu ihrem CO2-Ausstoss": "People pay proportionally to their CO2 emissions",
    "Le persone pagano in proporzione alle loro emissioni di CO2": "People pay proportionally to their CO2 emissions",

    "Les entreprises paient proportionnellement à leurs émissions de CO2": "Companies pay proportionally to their CO2 emissions",
    "Unternehmen zahlen proportional zu ihrem CO2-Ausstoss": "Companies pay proportionally to their CO2 emissions",
    "Le aziende pagano in proporzione alle loro emissioni di CO2": "Companies pay proportionally to their CO2 emissions",

    # Exemptions
    "Les personnes à faible revenu peuvent être exemptées des coûts": "Low-income earners exempted from costs",
    "mit Ausnahme von Menschen mit niedrigem Einkommen": "Low-income earners exempted from costs",
    "Le persone a basso reddito sono esentate dai costi": "Low-income earners exempted from costs",

    "Les personnes à faibles et moyens revenus peuvent être exemptées des coûts": "Low- and middle-income earners exempted from costs",
    "mit Ausnahme von Menschen mit niedrigem und mittlerem Einkommen": "Low- and middle-income earners exempted from costs",
    "Le persone a basso e medio reddito sono esentate dai costi": "Low- and middle-income earners exempted from costs",

    "Aucun groupe n'est exempté des coûts": "No groups exempted from costs",
    "Keine Gruppen sind von den Kosten ausgenommen": "No groups exempted from costs",
    "Nessun gruppo è esentato dai costi": "No groups exempted from costs",

    # Benefits
    "Les municipalités les plus touchées par les risques naturels, même si elles sont en déclin économique": "Municipalities most affected by natural hazards even if they are economically declining",
    "Gemeinden, die am stärksten von Naturgefahren betroffen sind, selbst wenn sie wirtschaftlich im Rückgang sind": "Municipalities most affected by natural hazards even if they are economically declining",
    "I comuni più a rischio dai pericoli naturali, anche se sono economicamente in declino": "Municipalities most affected by natural hazards even if they are economically declining",

    "Les municipalités économiquement prospères": "Economically prosperous municipalities",
    "Wirtschaftlich wohlhabende Gemeinden": "Economically prosperous municipalities",
    "I comuni economicamente prosperi": "Economically prosperous municipalities",

    "Les communes dans lesquelles les gens vivent depuis de nombreuses années doivent être protégées à tout prix": "Municipalities in which people have lived in for many years should be protected at all costs",
    "Gemeinden, in denen Menschen seit vielen Jahren leben, sollten um jeden Preis geschützt werden": "Municipalities in which people have lived in for many years should be protected at all costs",
    "I comuni in cui le persone vivono da molti anni devono essere protetti ad ogni costo": "Municipalities in which people have lived in for many years should be protected at all costs",

    "Niveaux de protection égaux pour toutes les municipalités": "Equal protection levels for all municipalities",
    "Gleiche Schutzniveaus für alle Gemeinden": "Equal protection levels for all municipalities",
    "Livelli di protezione uguali per tutti i comuni": "Equal protection levels for all municipalities",

    "Municipalités ayant une grande valeur culturelle, par exemple celles dotées de bâtiments historiques": "Culturally valuable municipalities e.g. with historic buildings",
    "Gemeinden mit vielen Kulturgütern wie z.B. historischen Gebäuden": "Culturally valuable municipalities e.g. with historic buildings",
    "Comuni di grande valore culturale, ad esempio con edifici storici": "Culturally valuable municipalities e.g. with historic buildings",
}

ANONYMIZE_COLS = ['Status', 'IPAddress', 'RecipientLastName','RecipientFirstName', 'RecipientEmail'
                , 'ExternalReference', 'LocationLatitude', 'LocationLongitude', 'consent_choice'
                , 'DistributionChannel', 'UserLanguage', r'language$', 'RecordedDate', 'ResponseId']

PREFERENCE_MAP = {"Option 1": 1, "Option 2": 2}

DONATIONS = [
    "I have donated money to support people in Blatten.",
    "I have donated money to another charity organization (not connected to Blatten).",
    "I have volunteered my time to support people in Blatten.",
    "I have volunteered my time to another charity organization (not connected to Blatten)."
]
