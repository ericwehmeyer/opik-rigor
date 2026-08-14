# Rubric: faithful summarisation of a source document

You are grading a summary against the source document it was written from. The
summary is intended for a reader who will act on it without opening the source.

## Scope

Grade only the summary's relationship to the source. Do not reward or penalise
tone, register, or house style, and do not check the source's own claims against
the world — a faithful summary of a wrong document is still a faithful summary.

## Criteria, in priority order

1. **Faithfulness (blocking).** Every claim in the summary must be supported by
   the source. A number, name, date, quantity, or causal link that the source
   does not state — or states differently — fails the summary outright, however
   good the rest of it is. Hedges the source did not make ("may", "roughly")
   count as changed claims when the source was definite, and so does dropping a
   hedge the source did make.
2. **Coverage of the load-bearing points.** A reader who acts on the summary
   alone must not be ambushed by something prominent in the source: the outcome
   or recommendation, its main stated reason, and any stated caveat, limit,
   deadline, or cost. Omitting decorative detail is correct; omitting a caveat is
   not.
3. **Attribution.** Where the source attributes a claim to someone, or marks it
   as disputed, provisional, or forecast, the summary must not restate it as
   settled fact.
4. **Self-containment.** The summary must be readable without the source: no
   dangling pronouns or references to "the above", "the table", or "as
   mentioned".
5. **Concision.** No padding, no restatement of the same point twice, no preamble
   about what the summary is about to do.

## Score

- **5** — Faithful, complete on every load-bearing point, correctly attributed,
  and tight. A reader could act on it.
- **4** — Faithful and self-contained; a minor secondary point is missing or the
  wording is looser than it needs to be. Still safe to act on.
- **3** — Faithful, but a reader would be missing something they would want: a
  stated caveat is thin, or attribution is vague. Usable only alongside the
  source.
- **2** — No outright fabrication, but a load-bearing point is missing or a
  disputed claim is presented as settled. Acting on this summary is risky.
- **1** — Contains a claim the source does not support, or inverts one it makes.

## Pass

Pass if the score is 4 or 5. A summary that fails criterion 1 always fails,
regardless of its other qualities.

## Output format

Nothing goes here, on purpose. `PinnedJudge` appends the response-format
instruction to the prompt itself — `PROMPT_TEMPLATE` ends in `{output_format}` —
so a rubric that restates it puts the same block in the rendered prompt twice.
If you want to read the exact wording the judge model receives, it is
`opik_rigor.judge.OUTPUT_FORMAT_INSTRUCTION`.

Write your own rubric the way this one is written: criteria and a scale, and
nothing about JSON.
