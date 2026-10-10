# The second brain (`lookup`, #374)

"What's the idea behind the Sicilian?" used to have nothing behind it but the
12B's memory, which knows some of it and makes up the rest. Since #374 the
planner has one more tool, `lookup(query)`, and the answer comes from a local
corpus of short notes. The narrator puts it in Glitch's words. Code finds the
notes; the model decides when to look something up and how to say it.

## The shape

```
player: "what's the idea behind the Sicilian?"
  → planner calls lookup(query="Sicilian Defense main idea")
  → knowledge.lookup searches the notes (BM25, CPU, a few milliseconds)
  → result: {"ok": true, "passages": [{"topic", "text"}, ...]}  (at most 3)
  → handoff sorts it as consulted; the narrator speaks from the passages
```

- **One tool, sources behind it.** `lookup` takes only `query`. There is no
  `source` or `topic` argument while there is one source: it would be a
  choice the 12B can only get wrong, and gemma-4-12b is schema-sensitive
  (`docs/agent-evals.md`, "Standing results"). Sources are a detail of
  `knowledge.py`; player memory (#377) would be the next source, not the
  next tool.
- **The first source is `chess_knowledge`:** the notes under
  `backend/src/chessapp/data/knowledge/`, one Markdown file per topic:
  `openings`, `strategy`, `tactics`, `endgames`, `rules`, `history` and
  `terms`, about 330 notes, each 20 to 130 words.
- **Placement.** `lookup` is offered last before `ask_player`, so every schema
  the planner was offered before it keeps its byte position and its cached
  prefix (#362). It is a read (`handoff.READ_TOOLS`) and is offered on the MCP
  surface too.
- **Nothing found is an answer.** A query that matches no note well enough
  returns `passages: []` and a `summary` saying so, and Glitch can say he
  doesn't know. A query with no meaningful words at all ("what is it?") is
  refused with `retry: different_args`.

## The notes

Written for this app from web research, in its own words: no copied text, so
no licence obligations beyond the Lichess opening names (CC0), which the
opening notes reuse. Each file opens with the references its facts were
checked against, so a wrong claim can be traced and fixed.

The format, per note:

```markdown
## Sicilian Defense: Najdorf Variation
Also: Najdorf; Sicilian Najdorf; 5...a6
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6. Named after ...
```

The heading is the title, the optional `Also:` line lists other names
(separated by `;`), and the paragraphs after it are what the narrator reads.

Opening notes are titled with the book's own names (`openings.book`), the
names the state block shows the planner (#373), so a planner that copies
"Ruy Lopez: Morphy Defense" from its state finds that note word for word.

Facts about current events (the world champion, the 2026 title match) are
dated in the text ("as of October 2026") and will go stale; update them in
place.

## Search

`knowledge.Index` is BM25 over the notes, with title and aliases weighted
three times over the text. Words are lower-cased, accent-stripped, lightly
stemmed ("castling" meets "castle", "pinned" meets "pin") and stopped
(question words and fillers). On top of the ranking, a note's title or alias
found word for word in the query earns a bonus weighted by how rare its words
are, so "Sicilian Najdorf" ranks the Najdorf note above the Sicilian one, and
a common-word alias earns almost nothing. "Defense" and "Variation" may be
left out when a name is said ("Caro-Kann Advance").

A result is at most `MAX_PASSAGES` (3) notes, none below `MIN_SCORE` and none
under half the best score. This is what `lookup` runs, and it's the fallback
whenever the embeddings service is down.

## Hybrid search (#450)

Keyword search misses reworded asks ("one piece attacks two of mine at once"
shares no word with *Fork*). It also injects on game chatter: "play e4" names
the King's Pawn Game note. `Index.hybrid` adds meaning. It's built for the
gather step (#448, #451), which searches the player's own words on every
turn. `lookup` stays on keywords until #451.

- **Vectors:** EmbeddingGemma 2 from the shared CPU service at
  `../embeddings` (port 8600, #449), through `chessapp/embeddings.py`. Plain
  httpx, the OpenAI `/v1/embeddings` wire, and EmbeddingGemma's task
  prefixes. A note is embedded with its title, its `Also:` names and its
  text (`knowledge.note_document`); adding the names lifted embedding recall
  by 10 rows. Note vectors are cached in one JSON file
  (`knowledge.NoteVectors`), keyed by the model and a hash of each note's
  document, because embedding all of them takes ~20 s. A query costs ~10 ms
  on the service, plus ~9 ms for the ranking.
- **Ranking:** reciprocal rank fusion (`RRF_K` 60) of the keyword ranking and
  the cosine ranking.
- **The bar** uses absolute signals, never the fused rank, because every
  query has a nearest note. Cosine alone can't be the bar: "pawn to d4" is
  0.82 from the Queen's Pawn Game note, while a real paraphrase can sit at
  0.67. What separates them is **lift**: a note's cosine above the query's
  11th-best note (`LIFT_BASELINE`). A note clears the bar in one of two ways:
  - it lifts `LIFT_ALONE` (0.065) on meaning alone, or
  - it lifts `LIFT_SUPPORT` (0.015) and scores `BM25_SUPPORT` (8) on keywords.

  Keywords alone no longer clear it.
- **This opening:** given the book name of the opening on the board, notes in
  its family gain about one extra first place (`OPENING_BOOST`), so "the
  advance variation" is the French one in a French game. The boost only
  reorders notes that already cleared the bar.
- **Vector size:** 768 dimensions. Matryoshka cuts measured worse (recall@3
  of 160 at 768, 156 at 512, 155 at 256), so nothing is truncated.

**Topical commands are allowed through.** "I resign", "offer a draw",
"castle kingside", "castle", "that was a blunder" and "is that checkmate"
are, to retrieval, the questions about those topics. No bar keeps them out
without dropping almost every real match (recall@3 would fall to ~15), so
they're reported, not pinned (decided 2026-10-10). The planner still does
what was asked. #451's eval gate watches that the note doesn't derail the
speech.

**Known misses** (the cost of zero false injections): "why does everyone say
knights belong in the middle" finds nothing. "Should I have accepted the
gambit" in a Queen's Gambit Accepted game finds the gambits and King's Gambit
Accepted notes, because the QGA note doesn't clear the bar.

Measured 2026-10-10 on the tables in `tests/knowledge_tables.py` (ANSWERS:
planner-style queries; SAID: player wording away from titles and aliases;
CHATTER: 42 in-game asks; TOPICAL: the six above).

| | ANSWERS r@1 / r@3 | SAID r@1 / r@3 | CHATTER found | TOPICAL found |
|---|---|---|---|---|
| keywords (BM25) | 120 / 120 of 120 | 27 / 36 of 63 | 9 / 42 | 6 / 6 |
| embeddings, lift bar | 110 / 112 | 23 / 25 | 0 / 42 | 5 / 6 |
| **hybrid** | 115 / 120 | 32 / 40 | **0 / 42** | 6 / 6 |

Hybrid finds as much as keywords, plus four reworded asks, and is the only
method that finds nothing on chatter. It sometimes puts a sibling note first
for an exact name (115 vs 120 first places), with the right one still in
the three.

## Tests

`tests/test_knowledge.py`, deterministic and in CI:

- **The retrieval table:** about 120 player questions, each with the note
  that must come first, across every file, plus asks no note should answer
  ("what time is it", "nice move"). A wording that stops finding its note is
  a ranking regression caught without a model.
- **The corpus:** every file names its sources; every note is short enough
  to read out; opening titles are book names; every move line in a note is
  legal; and an opening note's moves reach a position the book files under
  the note's own family.
- **The tool:** passages, the empty result, the refusals, and that a lookup
  moves nothing and reads as consulted in the handoff.
- **Hybrid** (`tests/test_knowledge.py`, `tests/test_embeddings.py`), on a
  pinned vector fixture, so the model never runs in CI. Every CHATTER ask
  finds nothing; hybrid recall@3 is at least the keywords' and at least the
  calibrated floor (160); the opening on the board ranks its own note first.
  `test_retrieval_report` prints the table above (`pytest -s`). The client
  tests cover the prefixes, batching, and every failure turning into
  `EmbeddingsUnavailable`. The cache tests cover embedding once, re-embedding
  an edited note, and a model change.

The tables live in `tests/knowledge_tables.py`. **After editing a note or
adding a query, regenerate the fixture** against the live service (the
fixture test fails and says so until you do):

```bash
cd backend
python scripts/embed_fixture.py      # writes tests/fixtures/knowledge_vectors.json
```

To check a wording by hand:

```bash
cd backend
python -c "from chessapp import knowledge as k; \
  print([(h.note.id, round(h.score, 1)) for h in k.chess_knowledge().search('your question')])"
```

## Speech accuracy

A lookup turn quotes the notes, not the board: "the Ruy Lopez is all about
Bb5" in a game that is in the Queen's Gambit is true. So the scorer's widened
facts (`speech_accuracy._widened`) back the moves a passage names and an
opening named from it, on that turn only (`docs/speech-accuracy.md`). A move
the notes never named stays unbacked.

## Measurement

Four frontier scenarios (`tests/frontier_corpus.py`):

- `knowledge_question`: a question about chess, graded on the note the
  lookup found.
- `this_openings_ideas`: "this opening" names nothing in the words; the
  planner must carry the state block's opening name into the query.
- `not_a_lookup`: near misses ("the best move in this position") that belong
  to `get_best_moves`, and must not draw a lookup.
- `knowledge_aside_then_move`: a lookup mid-game moves nothing, and the move
  after it lands.

A new tool changes what the planner is offered, so the change runs the eval
gate (`long_capture` ×3 is release-blocking) before merge, and the latency of
a lookup turn is read off the trace (`scripts/latency_report.py`).

## Decided, and why

- **This game's facts stay in context** (#373's opening and tally,
  `review_game`, `describe_position`). Moving them behind `lookup` would add a
  planner round trip for no gain. Revisit only with a measurement that says
  otherwise.
- **`lookup` is not an analysis tool.** It does not flip the narrator's
  thinking on and is not counted against the analysis budget: it costs
  milliseconds, not a Stockfish search.
