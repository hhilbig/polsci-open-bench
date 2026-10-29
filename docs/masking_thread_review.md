# Coherence review of the thread: proposals

A cold reader's suggestions for `docs/masking_twitter_thread.md`. None is
applied. Write yes, no, or your own wording on each HH line.

The reader took away this point: names make AI ratings more partisan, masking
removes that, but masked ratings match human coders less well, so report both.
The weak spot is post 3, where a reader thinks "wait, so which is it?"

## A. Post 2: "much less the case" reads as "no effect"

Current: "For others, this is much less the case: showing the names of parties
or politicians does not make ratings more partisan by a consistent amount."

Proposed: "For others, the names push them apart much less."

Why: every model moved (6 to 23 points); "much less" still says the size
varies.

**HH:** Yep, sure. 

## B. Post 3: say why masked ratings agree less with human coders (most important)

Current: "Without names, every model's ratings of individual tweets agree less
with human coders."

Proposed: "Without names, every model's ratings of individual tweets agree less
with human coders, who saw the names."

Why: post 1 shows the masked rating landing on the human rating; post 3 then
says masking makes models agree less with humans, without the reason.

**HH:** Yeah, sure. 

## C. Post 3: "overcorrect"

Current: "and some models overcorrect."

Proposed: "and some models put them closer together than human coders did."

Why: "overcorrect" leaves the direction for the reader to work out, and it
assumes a correct target, which the next paragraph questions.

**HH:** Sure. 

## D. Post 4: "worked about as well as masking"

Current: "telling models to simply ignore names worked about as well as
masking"

Proposed: "telling models to simply ignore names removed most of the extra
distance, as masking did"

Why: "worked" treats masking as the goal right after post 3 said that is
unclear.

**HH:** Sure. 

## E. Post 5: why switch from US tweets to UK manifestos

Current: "A 2016 study had experts rate UK manifesto sentences twice: first
with the manifesto title shown"

Proposed: "The tweet coders always saw the names, so I used a 2016 study in
which experts rated UK manifesto sentences twice: first with the party's
manifesto title shown"

Why: the switch isn't explained, and "manifesto title" doesn't obviously mean
the party is visible.

**HH:** Sure. 

## F. Post 5: three models unaccounted for

Current: "Only GPT-6.1 Sol and Gemini barely responded."

Proposed: "Two, GPT-6.1 Sol and Gemini, barely responded, and the rest
responded more."

Why: six matched the experts and two barely responded, which leaves three.
"Only" also suggests the other nine matched the experts, which conflicts with
"several times as much" in the next paragraph.

**HH:** Sure. 

## G. Post 6: "interesting"

Current: "raises an interesting question"

Proposed: "raises a question"

Why: this is the thread's main question, so state it plainly. You kept
"interesting" last time; listed only for completeness.

**HH:** No, we can leave it. 

## H. Post 7: "small benchmark"

Current: "it makes sense to add a small benchmark that shows the main results
with names shown and with names hidden"

Proposed: "it makes sense to report the main results twice, once with names
shown and once with names hidden"

Why: to this audience, "benchmark" suggests a separate dataset or leaderboard,
not a check of the main results.

**HH:** I want to say something like: at least some robustness, where some subset of statements is coded using names hidden to see what it looks like or something. 

## I. Post 7: give the case for masking too

Current: the second paragraph gives only the case for keeping names.

Proposed addition, at the end of that paragraph: "If you want to measure only
what the text says, masking seems the better choice."

Why: as written, the advice reads as "keep names" rather than "decide which
one you want", so "what should I do?" is only half answered.

**HH:** Sure. I don't want to use you. I want to use the researcher, or whatever. Yeah, the researcher is fine. 

## Left as is

- Post 1, "almost exactly where human coders put it": matches the figure (74
  vs 75). B resolves the tension it creates.
- Post 2, "further apart": slightly abstract, but figure 1's axis explains it.
- Post 5, "If we take changed ratings due to showing or hiding party labels as
  a natural part of how humans perceive political statements": wordy but reads
  correctly.
- "GPT-5.6" (post 4) and "GPT-6.1 Sol" (post 5) can be mistaken for the same
  model; both names are accurate.
