# Draft thread: party names and AI ratings of political text

Version 9 of the thread: Hanno's edits of 2026-10-19 with small fixes for
grammar and consistency only. Each fix is listed under its post as "Claude
fixed"; "Claude" notes flag points to check. No 280-character limit.

Figures are in `llm-party-names/figures/` (script `code/03_figures.R`); numbers
come from `llm-party-names/results/`, listed in the table at the end.

## Post 1

Attach: `02_example_tweets.png` (Republican grocery tweet and Democratic
McCarthy tweet; one mark per row)

> Some thoughts and evidence on using AI to tell whether politicians' statements are left- or right-leaning.
>
> How AI models classify political statements as left- or right-leaning depends a lot on whether the statement names a party.
>
> For example, a Republican tweet blames "Democrats' failed economic policies." Human coders rated it, on average, 75 on a scale from 0 to 100, where 100 is most right-wing. AI models rated it 88. If we give the same tweet to the same models but replace "Democrats" with [PARTY], they rate it almost exactly where human coders put it.
>
> github.com/hhilbig/llm-party-names

**HH:**

## Post 2

Attach: `05_names_push_apart.png` (title: "What happens if we hide or show names
in political statements?")

> I asked 11 AI models to rate tweets by members of Congress that human coders had already rated. Tweets that name a party or politician were rated twice: once as written, and once with the names masked as [PARTY] or [POLITICIAN].
>
> For some models, the names make Republican and Democratic tweets look much further apart on the left-right scale. For others, the names push them apart much less.

Claude fixed: "asked 11 AI models rate" → "to rate". In the clause after
the dash, "masking parties of politicians does not make ratings more partisan"
said the reverse of the result (showing the names makes ratings more partisan),
so it now reads "showing the names of parties or politicians does not make
ratings more partisan by a consistent amount".

**HH:** I added this last point because I want to say the magnitude with which they change the ratings is variable. 

## Post 3

Attach: `01_tweets_excess_separation.png` (four panels: commercial and
open-weight models in rows, names shown and names masked in columns)

> How does this compare to human coders who also saw the names? With names, six of the eleven models put Republicans' and Democrats' tweets further apart than human coders did. With names masked, that extra distance disappears, and some models put them closer together than human coders did.
>
> However, this doesn't necessarily mean the masked version is right. Without names, every model's ratings of individual tweets agree less with human coders, who saw the names. Masking clearly changes what the models do, but it's not obvious to me whether masked or unmasked ratings are more desirable.

Claude fixed: added the missing full stop at the end.

**HH:**  Small edit 

## Post 4

Attach: `04_ignore_instruction.png`

> Vallejo Vera and Driggers (2025) found that adding party labels to statements changed AI ratings. They also found that telling the model to ignore the labels doesn't fix the shift. I tried the same with names in tweets.
>
> For some models, like GPT-5.6 and Gemini, telling models to simply ignore names removed most of the extra distance, as masking did. Others, like Claude and DeepSeek, reacted much less to this additional instruction. GPT-6.1 Sol went the other way: told to ignore names, it put Republican and Democratic tweets much closer together than human coders did. Instructions such as "ignore the names" work for some models and not others, so they are not a reliable fix.

Claude fixed: Bluesky handles → "Vallejo Vera and Driggers (2025)", as you
said. "doesn't fix name-influenced ratings" → "doesn't fix the shift", because
their study added party labels; it did not test names in the text. "this
additional prompt" → "this additional instruction".

**HH:** For the posts, we're just not going to have the handles. We're just going to quote this in an academic way, like last name and last name (and then parentheses). 

## Post 5

Attach: `06_party_label_vs_experts.png`

> The clear next question is whether showing or hiding party labels makes as much of a difference to AI models as to humans. The tweet coders always saw the names, so I used a 2016 study by @kenbenoit and colleagues in which experts rated UK manifesto sentences twice: first with the party's manifesto title shown and later without it. I did the same with the models. Knowing the party moved six of the eleven models about as much as the experts. Two, GPT-6.1 Sol and Gemini, barely responded, and the rest responded more.
>
> If we take changed ratings due to showing or hiding party labels as a natural part of how humans perceive political statements, some AI models show somewhat similar behavior. However, AI models vary much more in how strongly they respond: from barely at all to several times as much as the experts.


Claude fixed:
1. Added the ">" markers so the text counts as post text.
2. "masking or unmasking names / parties" → "showing or hiding party labels":
   the expert study tests labels (the manifesto title) only.
3. "LLMs" → "AI models", the term used in every other post.
4. "GPT-6.1, Sol, and Gemini" → "GPT-6.1 Sol and Gemini": GPT-6.1 Sol is one
   model, so the commas made it read as three.
5. "as much as the experts" → "about as much as the experts": the six models
   are not distinguishable from the experts, which is not the same as equal.
6. Last sentence: "the range ... is much wider than that of human experts" →
   "AI models vary much more in how strongly they respond: from barely at all
   to several times as much as the experts." There is one expert estimate,
   not a range across experts, so the comparison as written didn't hold.

Claude: to avoid repeating this point, post 6 no longer has the sentence
"Validation against human coders has the same issue: they react to party labels
too." This post makes that point.

**HH:** 

## Post 6

> Using AI to read political texts raises an interesting question: should the model use the names or not? The names carry information that people themselves use when rating these statements, and the same statement by a Republican or a Democratic politician may convey a different meaning.

**HH:**

## Post 7

> Others have recommended masking names (Walker and Angst 2025; Elbouanani et al. 2025). However, humans respond to party labels too.
>
> Whether to mask names depends on how much researchers think the speaker changes the meaning of a statement. For researchers who study how people perceive or respond to political statements, for example as a treatment in studies of attitudes or behavior, keeping the names seems the better choice, since people see who is speaking. If researchers want to measure only what the text says, masking seems the better choice.
>
> In any event, it makes sense to run at least a robustness check in which a subset of statements is coded with names hidden, so the reader can see how much of a difference this makes for the analysis.

Claude fixed: ended the last sentence with a full stop instead of a comma,
and rewrote it for grammar: "a small benchmark that shows main results based on
labels that were derived from masking (with names hidden and shown) makes
sense" → "it makes sense to add a small benchmark that shows the main results
with names shown and with names hidden"; "for what the researcher is doing" →
"for the analysis".

Claude fixed: "humans clearly respond to the presence of names or party
labels" → "humans respond to party labels too", as you agreed.

**HH:**

## Sources

| Claim | Source |
|---|---|
| Grocery tweet 75 / 88 / 74; McCarthy tweet 35 / 14 / 26 | `results/tweet_items.csv`, items `lemens_tweet_226` and `lemens_tweet_183`; mean of Gemini 3.8 Flash, DeepSeek V4 Pro, GPT-5.6 Terra, Claude Sonnet 5 |
| Names push the parties apart by 6 to 23 points | `results/tweet_masking.csv`, named rows, `rep_minus_dem_change`: −6.3 (Gemini) to −22.9 (Qwen3 30B); all intervals exclude zero |
| 6 of 11 further apart than people, up to 13 points | `rep_dem_error_gap_orig` with intervals above zero: Gemini 9.0, Gemma 4 9.7, Qwen3 30B 10.1, Claude 11.0, DeepSeek 11.8, GPT-5.6 12.8 |
| Every model agrees less with people without names | Mean absolute error against raters, named tweets: higher without names in all 11 models, by 1.1 to 5.3 points |
| Most models as much as experts; Sol and Gemini barely; three open models 2 to 4 times | `results/title_effects.csv`, rows `common` (1,033 sentences): experts 0.22 [0.12, 0.35] steps; Sol and Gemini 0.04; Qwen3.6 35B 0.39, Nemotron 0.50, Qwen3 30B 0.90 |
| Sol and Gemini move least with names in tweets | `results/tweet_masking.csv`, named rows: Gemini −6.3, Sol −7.8, the two smallest of 11 |
| 144 named tweets; commercial models rated 233 tweets, open-weight 500 | `results/tweet_masking.csv`, `n`: named 144 for all models; unnamed 89 (commercial) and 356 (open-weight) |
| With names masked, commercial models near people, open-weight models below | `rep_dem_error_gap_mask`: commercial −1.8 to 2.9 (Sol −5.1); open-weight −5.1 (Gemma) to −24.2 (Nemotron) |
| Tweets that name no one barely moved | `results/tweet_masking.csv`, placebo rows, `rep_minus_dem_change`: −0.8 to 1.5 points across all 11 models |
| Prior work recommends masking | Walker & Angst (2025, PLOS ONE 20(11): e0335547) recommend masking entity names (organizations, as "Entity A") because the cost is small; Elbouanani et al. (2025, Findings of ACL) replace politician names with fictional ones to reduce bias. Neither says "always". |
| Ignore instruction | `results/ignore_instruction_tweets.csv`, extra distance as written → told to ignore (names masked): Gemini 9.0 → 1.2 (2.7), GPT-5.6 12.8 → 3.4 (2.9), Claude 11.0 → 7.6 (0.9), DeepSeek 11.8 → 9.4 (−1.8), GPT-6.1 Sol 2.8 → −9.6 (−5.1) |

The expert comparison is not randomized: experts saw the title on the first
pass and coded again without it about a year later, so their effect also
contains coding order and time. On all 3,665 sentences (open models only) the
expert effect is 0.12, and all six open models react more than experts.
