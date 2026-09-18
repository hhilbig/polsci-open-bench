CAP_MAJOR_TOPIC_LABELS = {
    1: "Macroeconomics",
    2: "Civil Rights",
    3: "Health",
    4: "Agriculture",
    5: "Labor and Immigration",
    6: "Education",
    7: "Environment",
    8: "Energy",
    9: "Immigration",
    10: "Transportation",
    12: "Law and Crime",
    13: "Social Welfare",
    14: "Housing",
    15: "Domestic Commerce",
    16: "Defense",
    17: "Technology",
    18: "Foreign Trade",
    19: "International Affairs",
    20: "Government Operations",
    21: "Public Lands",
    23: "Culture",
}

CAP_MAJOR_TOPIC_LABELS_IN_ORDER = list(CAP_MAJOR_TOPIC_LABELS.values())

# v2 naming. Major topic 5 was carried over under its legacy name "Labor and
# Immigration", which predates the split of immigration into its own major topic
# 9. Keeping both in one label list gave the model two plausible buckets for the
# same content: in the v1 run, gold "Labor and Immigration" drew 37 "Social
# Welfare" and 10 "Civil Rights" predictions, and 5 of the gold "Immigration"
# items were answered "Labor and Immigration". The current CAP master codebook
# calls topic 5 Labor.
#
# v1 manifests keep the old name so published results stay reproducible; v2
# manifests apply CAP_MAJOR_TOPIC_RENAMES_V2 through ground_truth.label_map,
# which renames gold values at load time without regenerating the cleaned CSVs.
CAP_MAJOR_TOPIC_LABELS_V2 = {**CAP_MAJOR_TOPIC_LABELS, 5: "Labor"}

CAP_MAJOR_TOPIC_LABELS_V2_IN_ORDER = list(CAP_MAJOR_TOPIC_LABELS_V2.values())

CAP_MAJOR_TOPIC_RENAMES_V2 = {
    old: CAP_MAJOR_TOPIC_LABELS_V2[code]
    for code, old in CAP_MAJOR_TOPIC_LABELS.items()
    if old != CAP_MAJOR_TOPIC_LABELS_V2[code]
}
