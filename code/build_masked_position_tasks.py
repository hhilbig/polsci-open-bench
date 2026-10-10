#!/usr/bin/env python3
"""
Build party-masked copies of three ideological-position extension tasks.

Question: do models place a statement further right when the passage names a
right-wing party or politician, independently of what the statement says? The
Benoit et al. (2016) experts coded sentences without knowing their source, but
the passage shown to the model names a party in about 40% of Conservative and
Labour economic sentences, and the Conservative-minus-Labour gap in model error
appears only in those sentences (code/analyze_task_ext_position_validity.py).

Each masked task keeps the rows and row order of the original data file, so the
benchmark's seeded 250+250 draw selects exactly the same 500 items, and the
predictions can be compared item by item. Only the text columns change:

- party names -> "[PARTY]" (Labour, New Labour, Conservative(s), Tory/Tories,
  Liberal Democrat(s), Lib Dem(s), capitalised Liberal(s), the 1987 Alliance,
  SDP, Social Democrats; Democrat(s)/Democratic Party, Dem(s), Republican(s),
  GOP, MAGA in tweets)
- named politicians -> "[POLITICIAN]" (full names, "Mr/Mrs" + surname, and
  bare surnames that are unambiguous in these texts; politician and party
  Twitter handles; hashtags containing a name, e.g. #BidenFlation)

Kept as content: ideology words in lower case ("conservative values"),
"Socialist Spain", the government/opposition role, slogans and policy terms.
"British Steel" and "Foot and Mouth" are not names.

Writes data/<task>_masked.csv and tasks_ext_masked/<task>_masked.yaml, and
prints how many items in each 500-item sample changed.
"""
import re
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
TASKS = {
    "benoit_manifesto_economic_position": ["context_before", "text", "context_after"],
    "benoit_manifesto_social_position": ["context_before", "text", "context_after"],
    "lemens_gallego_tweet_position": ["text"],
}
OUT_TASKS = REPO / "tasks_ext_masked"

UK_POLITICIANS = [
    r"(?:Mrs\.? |Margaret )Thatcher", r"Thatcher", r"(?:Mr\.? |John )Major", r"Major(?= (?:[Gg]overnments?|years))",
    r"(?:Mr\.? |Gordon )Brown", r"(?:Mr\.? |Tony )?Blair", r"(?:Mr\.? |Michael )Foot(?! and Mouth)",
    r"(?:Mr\.? |David )?Cameron", r"(?:Mr\.? |Neil )?Kinnock", r"(?:Mr\.? |Michael )Howard",
    r"(?:Mr\.? |David )Steel", r"(?:Mr\.? |David )Owen", r"(?:Mr\.? |Paddy )?Ashdown",
    r"(?:Mr\.? |Charles )Kennedy", r"(?:Mr\.? |Nick )?Clegg", r"(?:Mr\.? |William )?Hague",
    r"(?:Mr\.? |Iain )?Duncan Smith", r"(?:Mr\.? |John )Smith",
]
UK_PARTIES = [
    r"New Labour", r"Labour", r"Conservatives?", r"Tory", r"Tories",
    r"Liberal Democrats?", r"Lib Dems?", r"Liberals?(?= [Pp]art)", r"Liberals", r"Liberal(?= (?:insistence|devolution|policies))",
    r"(?<=\[PARTY\] and )Liberal",
    r"SDP", r"Social Democrats?", r"(?<![Aa]tlantic )Alliance",
]
US_POLITICIANS = [
    r"@\w*(?:Biden|Trump|McCarthy|Pelosi|Schumer|McConnell|Jeffries|Harris|Obama|DeSantis|POTUS|VP)\w*",
    r"#\w*(?:Biden|Trump|Pelosi|Schumer|McConnell|Obama|DeSantis)\w*",
    r"(?:President |Joe |Donald |Speaker |Kevin |Nancy |Leader )?"
    r"(?:Biden|BIDEN|Trump|McCarthy|Pelosi|Schumer|McConnell|Jeffries|Obama|DeSantis|Desantis)(?:nomics|'s)?",
    r"(?:Vice President |VP |Kamala )Harris",
]
US_PARTIES = [
    r"@\w*(?:GOP|Democrats|Dems|Republicans)\w*", r"#\w*(?:GOP|MAGA|Democrats|Republicans)\w*",
    r"Democratic Party", r"Republican Party", r"Democratic(?= Issues Conference| Caucus)",
    r"Republicanos", r"Democrats?", r"Dems?", r"Republicans?",
    r"republicans", r"GOP", r"MAGA",
]


def compile_rules(politicians, parties):
    rules = [(re.compile(r"(?<![\w@#])(?:%s)(?!\w)" % p), "[POLITICIAN]") for p in politicians]
    rules += [(re.compile(r"(?<![\w@#])(?:%s)(?!\w)" % p), "[PARTY]") for p in parties]
    # Handles and hashtags start with @ or #, so they are matched without the
    # leading word-boundary guard.
    rules = [(re.compile(r"(?:%s)" % p), "[POLITICIAN]") for p in politicians if p.startswith(("@", "#"))] + rules
    rules = [(re.compile(r"(?:%s)" % p), "[PARTY]") for p in parties if p.startswith(("@", "#"))] + rules
    return rules


def mask(text, rules):
    if not isinstance(text, str):
        return text
    for pattern, placeholder in rules:
        text = pattern.sub(placeholder, text)
    return text


def main():
    import task_registry  # noqa: E402  (imported here so the script runs from repo root)

    OUT_TASKS.mkdir(exist_ok=True)
    for task, columns in TASKS.items():
        rules = compile_rules(*(
            (UK_POLITICIANS, UK_PARTIES) if task.startswith("benoit") else (US_POLITICIANS, US_PARTIES)))
        data = pd.read_csv(REPO / "data" / f"{task}.csv")
        masked = data.copy()
        for col in columns:
            masked[col] = data[col].map(lambda t: mask(t, rules))
        changed = (masked[columns].fillna("") != data[columns].fillna("")).any(axis=1)
        masked["masked_changed"] = changed
        masked.to_csv(REPO / "data" / f"{task}_masked.csv", index=False)

        spec = yaml.safe_load((REPO / "tasks_ext" / f"{task}.yaml").read_text())
        spec["name"] = f"{task}_masked"
        spec["data_file"] = f"../data/{task}_masked.csv"
        spec["order"] = int(spec["order"]) + 100
        out = OUT_TASKS / f"{task}_masked.yaml"
        header = (f"# Party-masked copy of tasks_ext/{task}.yaml, built by\n"
                  f"# code/build_masked_position_tasks.py. Same rows, same 500-item draw.\n")
        out.write_text(header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

        orig = task_registry.load_task_definitions(task_manifest=REPO / "tasks_ext" / f"{task}.yaml")[0]
        new = task_registry.load_task_definitions(task_manifest=out)[0]
        a, b = orig["loader"](), new["loader"]()
        assert [x["item_id"] for x in a] == [x["item_id"] for x in b], "item draw changed"
        n_changed = sum(x["user_content"] != y["user_content"] for x, y in zip(a, b))
        print(f"{task}: {changed.sum()} of {len(data)} rows changed; "
              f"{n_changed} of {len(a)} sampled items changed")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO / "code"))
    main()
