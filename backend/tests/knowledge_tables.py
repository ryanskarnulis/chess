"""The retrieval tables for the second brain (#374, #450).

Shared by `test_knowledge.py` and `scripts/embed_fixture.py`, which embeds
every query here into the pinned vector fixture: a query added to a table
needs the fixture regenerated (the test that finds it missing says how).
"""

# What a player asks, as the planner would pass it on, and the note that
# answers it: the first hit, always.
ANSWERS = [
    # openings
    ("what's the idea behind the Sicilian", "openings/sicilian-defense"),
    ("Sicilian Najdorf", "openings/sicilian-defense-najdorf-variation"),
    ("Najdorf plans", "openings/sicilian-defense-najdorf-variation"),
    ("Sicilian Dragon Yugoslav attack", "openings/sicilian-defense-dragon-variation"),
    ("Sveshnikov", "openings/sicilian-defense-lasker-pelikan-variation"),
    ("Ruy Lopez: Morphy Defense", "openings/ruy-lopez-morphy-defense"),
    ("plans in the Ruy Lopez Morphy Defense", "openings/ruy-lopez-morphy-defense"),
    ("Spanish opening", "openings/ruy-lopez"),
    ("Berlin wall", "openings/ruy-lopez-berlin-defense"),
    ("Marshall Attack", "openings/ruy-lopez-marshall-attack"),
    ("Italian Game ideas", "openings/italian-game"),
    ("Giuoco Piano", "openings/italian-game"),
    (
        "Fried Liver Attack",
        "openings/italian-game-two-knights-defense-fried-liver-attack",
    ),
    ("Evans Gambit", "openings/italian-game-evans-gambit"),
    ("French Defence main ideas", "openings/french-defense"),
    ("French Winawer", "openings/french-defense-winawer-variation"),
    ("Advance French", "openings/french-defense-advance-variation"),
    ("Caro-Kann", "openings/caro-kann-defense"),
    ("Caro Kann advance variation", "openings/caro-kann-defense-advance-variation"),
    ("Scandinavian defense", "openings/scandinavian-defense"),
    ("how to play against the London System", "openings/london-system"),
    ("Queen's Gambit", "openings/queen-gambit"),
    ("queens gambit declined", "openings/queen-gambit-declined"),
    ("Queen's Gambit Accepted", "openings/queen-gambit-accepted"),
    ("Slav defense", "openings/slav-defense"),
    ("King's Indian Defense plans", "openings/king-indian-defense"),
    ("Nimzo-Indian", "openings/nimzo-indian-defense"),
    ("Grunfeld", "openings/grunfeld-defense"),
    ("Dutch Stonewall", "openings/dutch-defense"),
    ("Catalan", "openings/catalan-opening"),
    ("is the King's Gambit any good", "openings/king-gambit"),
    ("English opening", "openings/english-opening"),
    ("Bongcloud", "openings/bongcloud-attack"),
    ("Benko Gambit", "openings/benko-gambit"),
    ("Alekhine's defence", "openings/alekhine-defense"),
    ("Pirc", "openings/pirc-defense"),
    ("Smith-Morra gambit", "openings/sicilian-defense-smith-morra-gambit"),
    ("what opening should a beginner play", "strategy/choosing-an-opening"),
    ("what is a gambit", "strategy/gambits"),
    # strategy
    ("why is the bishop pair good", "strategy/the-bishop-pair"),
    ("opening principles", "strategy/opening-principles"),
    ("why castle early", "strategy/king-safety"),
    ("isolated queen's pawn", "strategy/isolated-queen-pawn"),
    ("what is an isolated pawn", "strategy/isolated-queen-pawn"),
    ("doubled pawns", "strategy/doubled-pawns"),
    ("passed pawn", "strategy/passed-pawn"),
    ("good bishop bad bishop", "strategy/good-and-bad-bishops"),
    ("knight on the rim is dim", "strategy/knights"),
    ("how much is a rook worth", "strategy/material-values"),
    ("rook on the seventh rank", "strategy/rook-on-the-seventh"),
    ("what does fianchetto mean", "strategy/fianchetto"),
    ("hypermodern", "strategy/hypermodernism"),
    ("prophylaxis", "strategy/prophylaxis"),
    ("when should I trade pieces", "strategy/when-to-trade-pieces"),
    ("how do I stop blundering", "strategy/blunder-check"),
    ("minority attack", "strategy/minority-attack"),
    ("outpost for a knight", "strategy/outposts"),
    ("how to improve at chess", "strategy/how-to-improve-at-chess"),
    # tactics
    ("what's a fork", "tactics/fork"),
    ("absolute pin", "tactics/pin"),
    ("skewer", "tactics/skewer"),
    ("discovered check", "tactics/discovered-check"),
    ("zwischenzug", "tactics/zwischenzug"),
    ("smothered mate", "tactics/smothered-mate"),
    ("back rank mate", "tactics/back-rank-mate"),
    ("Greek gift sacrifice", "tactics/greek-gift-sacrifice"),
    ("scholar's mate", "tactics/scholar-mate"),
    ("fastest checkmate", "tactics/fool-mate"),
    ("Legal's mate", "tactics/legal-mate"),
    ("Elephant trap", "tactics/elephant-trap"),
    ("underpromotion to a knight", "tactics/underpromotion"),
    # endgames
    ("opposition in king and pawn endings", "endgames/opposition"),
    ("how to checkmate with king and rook", "endgames/checkmate-with-king-and-rook"),
    ("checkmate with a queen", "endgames/checkmate-with-king-and-queen"),
    ("bishop and knight checkmate", "endgames/checkmate-with-bishop-and-knight"),
    ("can two knights checkmate", "endgames/two-knights-cannot-force-mate"),
    ("Lucena position", "endgames/lucena-position"),
    ("Philidor position", "endgames/philidor-position"),
    ("explain zugzwang", "endgames/zugzwang"),
    ("rule of the square", "endgames/rule-of-the-square"),
    ("triangulation", "endgames/triangulation"),
    ("opposite coloured bishops endgame", "endgames/opposite-coloured-bishop-endgames"),
    ("tablebases", "endgames/endgame-tablebases"),
    ("wrong rook pawn", "endgames/wrong-rook-pawn"),
    # rules
    ("how does en passant work", "rules/en-passant"),
    ("can I castle out of check", "rules/castling-edge-cases"),
    ("castling rules", "rules/castling"),
    ("what is the fifty move rule", "rules/fifty-move-rule"),
    ("50 move rule", "rules/fifty-move-rule"),
    ("threefold repetition", "rules/threefold-repetition"),
    ("what is stalemate", "rules/stalemate"),
    ("insufficient material", "rules/insufficient-material"),
    ("how do knights move", "rules/how-the-knight-moves"),
    ("can a pawn move backwards", "rules/how-the-pawn-moves"),
    ("can I have two queens", "rules/pawn-promotion"),
    ("touch move rule", "rules/touch-move"),
    ("what is an Elo rating", "rules/chess-ratings"),
    ("how do you become a grandmaster", "rules/chess-titles"),
    ("Chess960", "rules/chess960"),
    ("what is blitz", "rules/time-controls"),
    ("how to read chess notation", "rules/algebraic-notation"),
    # history
    ("who invented chess", "history/origins-of-chess"),
    ("how old is chess", "history/origins-of-chess"),
    ("when did the queen get so strong", "history/modern-chess-and-the-mad-queen"),
    ("when did castling start", "history/history-of-castling"),
    ("who was Capablanca", "history/jose-raul-capablanca"),
    ("tell me about Bobby Fischer", "history/bobby-fischer"),
    ("who is the current world champion", "history/current-world-champion"),
    ("Magnus Carlsen", "history/magnus-carlsen"),
    ("Deep Blue", "history/deep-blue"),
    ("the Immortal Game", "history/the-immortal-game"),
    ("Opera game", "history/the-opera-game"),
    ("first world chess champion", "history/wilhelm-steinitz"),
    ("strongest female chess player", "history/women-in-chess"),
    ("AlphaZero", "history/modern-chess-engines"),
    ("Stockfish", "history/modern-chess-engines"),
    # terms
    ("what does en prise mean", "terms/en-prise"),
    ("ECO codes", "terms/eco-codes"),
    ("what is a simul", "terms/simultaneous-exhibition"),
    ("bughouse", "terms/chess-boxing-and-other-variants"),
]

# Asks that are not about chess knowledge at all: nothing should come back.
NOTHING = [
    "pizza recipe",
    "what time is it",
    "how are you today",
    "nice move",
    "let's play",
    "hello",
    "thanks",
    "do it",
    "weather in Paris",
]

# What a player says out loud, in the middle of a game, and the note that
# answers it (#450). Worded away from the notes' titles and aliases on
# purpose: this is what keyword search misses and meaning should catch.
# Graded at recall@3, not first place.
SAID = [
    (
        "what do you call it when one piece attacks two of mine at the same time",
        "tactics/fork",
    ),
    ("why can't my piece move, it's stuck in front of my king", "tactics/pin"),
    (
        "what's it called when you check the king and win the piece behind it",
        "tactics/skewer",
    ),
    (
        "moving one piece out of the way so another one gives check",
        "tactics/discovered-check",
    ),
    ("checked by two pieces at the same time", "tactics/double-check"),
    (
        "making a surprise move in the middle of a trade instead of taking back",
        "tactics/zwischenzug",
    ),
    (
        "the mate where the king is boxed in by its own pieces and a knight hits it",
        "tactics/smothered-mate",
    ),
    (
        "my king got mated on the last row because my own pawns were in the way",
        "tactics/back-rank-mate",
    ),
    (
        "giving up the bishop on h7 to go after the castled king",
        "tactics/greek-gift-sacrifice",
    ),
    (
        "the four move checkmate beginners fall for with the queen and bishop on f7",
        "tactics/scholar-mate",
    ),
    ("turning a pawn into something other than a queen", "tactics/underpromotion"),
    ("one of my pieces is guarding too many things at once", "tactics/overloading"),
    ("checking over and over so the game ends in a draw", "tactics/perpetual-check"),
    ("giving up material on purpose to get something better", "tactics/sacrifice"),
    ("why do people say you should tuck your king away early", "strategy/king-safety"),
    ("what's so good about having both bishops", "strategy/the-bishop-pair"),
    ("how many points is each piece worth", "strategy/material-values"),
    (
        "two of my pawns ended up on the same file, is that bad",
        "strategy/doubled-pawns",
    ),
    ("a pawn with no enemy pawns in front of it or beside it", "strategy/passed-pawn"),
    ("should I put my rook on a file with no pawns on it", "strategy/open-files"),
    ("what should I be doing in the first few moves", "strategy/opening-principles"),
    ("is it good to swap off pieces", "strategy/when-to-trade-pieces"),
    ("I keep leaving my pieces where they can just be taken", "strategy/blunder-check"),
    ("I have no idea what to do in the middle of the game", "strategy/planning"),
    ("how do I get better at this game", "strategy/how-to-improve-at-chess"),
    ("a square the enemy pawns can never kick my knight from", "strategy/outposts"),
    ("putting the bishop on g2 behind the pawn on g3", "strategy/fianchetto"),
    ("stopping what my opponent wants to do before they do it", "strategy/prophylaxis"),
    ("which is stronger, a knight or a bishop", "strategy/bishop-versus-knight"),
    ("I'm losing, how do I hang on", "strategy/defending-a-worse-position"),
    ("I always run short on the clock", "strategy/time-management"),
    (
        "how do I mate with just a rook and my king",
        "endgames/checkmate-with-king-and-rook",
    ),
    ("the kings facing each other with one square between them", "endgames/opposition"),
    ("when you'd rather pass but every move makes it worse", "endgames/zugzwang"),
    ("can my king catch that pawn before it queens", "endgames/rule-of-the-square"),
    (
        "the computer databases that solved every position with a few pieces left",
        "endgames/endgame-tablebases",
    ),
    (
        "bishops on different colours are drawish, right",
        "endgames/opposite-coloured-bishop-endgames",
    ),
    ("can I win with just two knights", "endgames/two-knights-cannot-force-mate"),
    ("in the endgame should my king come out and fight", "endgames/active-king"),
    (
        "wait, can a pawn take another pawn that just went past it two squares",
        "rules/en-passant",
    ),
    ("how does the horse move again", "rules/how-the-knight-moves"),
    ("what happens when my pawn reaches the other side", "rules/pawn-promotion"),
    (
        "can I still swap my king and rook if I've been checked",
        "rules/castling-edge-cases",
    ),
    ("I'm not in check but nothing I have can move", "rules/stalemate"),
    ("if the same position keeps coming up is it a draw", "rules/threefold-repetition"),
    ("how many moves without a capture before it's a draw", "rules/fifty-move-rule"),
    ("if I touch a piece do I have to move it", "rules/touch-move"),
    ("how do the numbers in chess ratings work", "rules/chess-ratings"),
    ("the version where the back row pieces are shuffled", "rules/chess960"),
    ("how do I write the moves down", "rules/algebraic-notation"),
    (
        "how do you put the pieces on the board at the start",
        "rules/the-board-and-setup",
    ),
    ("where does chess come from", "history/origins-of-chess"),
    ("who holds the world title these days", "history/current-world-champion"),
    ("the computer that beat Kasparov in the nineties", "history/deep-blue"),
    ("the American who beat Spassky in 1972", "history/bobby-fischer"),
    (
        "that old chess machine that was really a person hiding inside",
        "history/the-turk",
    ),
    (
        "when did the queen become the strongest piece",
        "history/modern-chess-and-the-mad-queen",
    ),
    ("what's the point of playing c5 against e4", "openings/sicilian-defense"),
    (
        "what's black after when they play c6 and then d5 against e4",
        "openings/caro-kann-defense",
    ),
    (
        "white pushes d4 and c4 and offers a pawn, what's that about",
        "openings/queen-gambit",
    ),
    ("what's black trying to do with e6 and d5 against e4", "openings/french-defense"),
    ("is giving up the f pawn on move two any good for white", "openings/king-gambit"),
    ("what is white aiming for with that bishop to f4 setup", "openings/london-system"),
]

# The same words in different games: the opening on the board decides which
# note "the advance variation" is (`Index.hybrid`'s opening boost). Known
# limit, not pinned: "should I have accepted the gambit" in a Queen's Gambit
# Accepted game finds the gambits note and the King's Gambit Accepted one,
# because the QGA note doesn't clear the bar and the boost never lifts a note
# past it.
SAID_IN_OPENING = [
    (
        "what's the plan in the advance variation",
        "French Defense: Advance Variation",
        "openings/french-defense-advance-variation",
    ),
    (
        "what's the plan in the advance variation",
        "Caro-Kann Defense: Advance Variation",
        "openings/caro-kann-defense-advance-variation",
    ),
    (
        "what's the idea of the exchange variation",
        "Ruy Lopez: Exchange Variation",
        "openings/ruy-lopez-exchange-variation",
    ),
    (
        "what's the idea of the exchange variation",
        "French Defense: Exchange Variation",
        "openings/french-defense-exchange-variation",
    ),
    (
        "what's the idea of the exchange variation",
        "Queen's Gambit Declined: Exchange Variation",
        "openings/queen-gambit-declined-exchange-variation",
    ),
    (
        "the classical variation, what's it about",
        "French Defense: Classical Variation",
        "openings/french-defense-classical-variation",
    ),
    (
        "the classical variation, what's it about",
        "Caro-Kann Defense: Classical Variation",
        "openings/caro-kann-defense-classical-variation",
    ),
    (
        "should I have accepted the gambit",
        "King's Gambit Accepted",
        "openings/king-gambit-accepted",
    ),
]

# Said mid-game and about no topic in the notes: hybrid search must find
# nothing for any of them (a near-miss passage in context is noise for the
# planner and fuel for speech errors).
CHATTER = [
    *NOTHING,
    "play e4",
    "knight to f3",
    "bishop takes c6",
    "pawn to d4",
    "queen to h5",
    "I'll take the knight",
    "take it back",
    "undo that",
    "undo my last move",
    "can I take back my move",
    "good game",
    "what's the best move here",
    "what should I play now",
    "who's winning",
    "did I just lose my queen",
    "go easier on me",
    "make it harder",
    "hint please",
    "give me a hint",
    "new game",
    "let's play again",
    "I want to play black",
    "flip the board",
    "turn the voice off",
    "save the game",
    "what did you just play",
    "why did you play that",
    "oops",
    "wow",
    "your move",
    "hurry up",
    "are you there",
    "hmm let me think",
]

# Commands that name a note's topic word for word. To retrieval they are
# the question about that topic ("I resign" is the resignation note), and no
# bar tells them apart without losing nearly every real match, so they are
# reported, not pinned (decided 2026-10-10, #450): the note may come along,
# and the planner still does what was asked.
TOPICAL = [
    "I resign",
    "offer a draw",
    "castle kingside",
    "castle",
    "that was a blunder",
    "is that checkmate",
]
