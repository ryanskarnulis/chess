# Rules

Notes on the rules of chess: how the pieces move, the special moves, how a
game ends, and the rules of competitive play. Written for this app in its own
words. In this app the board itself enforces the rules; these notes explain
them.

Sources the facts were checked against: the FIDE Laws of Chess (in effect
from 1 January 2023), Wikipedia's articles on the rules of chess and the
individual rules, and the US Chess / FIDE rules comparison.

## The board and setup
Also: how to set up the board; starting position; light square on the right; queen on her colour
The board has 64 squares in eight files (a to h) and eight ranks (1 to 8).
Each player starts with sixteen pieces: a king, a queen, two rooks, two
bishops, two knights and eight pawns. A light square goes in each player's
bottom-right corner ("light on the right"). Rooks go in the corners, then
knights, then bishops; the queen goes on her own colour (white queen on d1,
black queen on d8) and the king beside her. White always moves first.

## How the king moves
Also: king moves; king move; can the king capture
The king moves one square in any direction. It may capture, but it may never
move onto a square attacked by an enemy piece, and two kings can never stand
next to each other. It also has the special move castling.

## How the queen moves
Also: queen moves; queen move
The queen moves any number of squares along a rank, file or diagonal, as long
as nothing is in the way: a rook and a bishop combined. It is the most
powerful piece.

## How the rook moves
Also: rook moves; rook move; castle piece
The rook moves any number of squares along a rank or file, without jumping.
It takes part in castling. Some people call the rook a "castle", but in chess
"castling" is the name of the special king move.

## How the bishop moves
Also: bishop moves; bishop move
The bishop moves any number of squares diagonally, without jumping. Each
bishop stays on squares of one colour for the whole game, so each side has a
light-squared and a dark-squared bishop.

## How the knight moves
Also: knight moves; knight move; L shape; horse
The knight moves in an L shape: two squares in one direction along a rank or
file, then one square at a right angle. It is the only piece that can jump
over other pieces. A knight always lands on a square of the opposite colour
to the one it started on.

## How the pawn moves
Also: pawn moves; pawn move; can a pawn move backwards; pawn first move two squares
A pawn moves straight forward one square, or two squares from its starting
square if both are empty. It captures differently: one square diagonally
forward. Pawns never move backward or sideways, and a pawn blocked by any
piece directly in front of it cannot move forward. Pawns have two special
moves: en passant and promotion.

## Pawn promotion
Also: promotion; promote; queening; can you have two queens
A pawn that reaches the far rank must be promoted, on the same move, to a
queen, rook, bishop or knight of its own colour. The choice is free and not
limited to captured pieces, so a player can have two or more queens. Most
players choose a queen; a different piece is called underpromotion.

## Castling
Also: castling rules; how to castle; O-O; O-O-O; castle kingside; castle queenside; when can you castle
Castling is a single move of the king and a rook: the king moves two squares
toward the rook, and the rook jumps to the square the king passed over.
Kingside castling (O-O) ends with the king on g1 and the rook on f1;
queenside (O-O-O) with the king on c1 and rook on d1 (the same on rank 8 for
Black). Conditions: neither the king nor that rook has moved before in the
game; the squares between them are empty; the king is not in check; and the
king does not pass through or land on a square attacked by the enemy.

## Castling edge cases
Also: castle out of check; castle through check; can the rook be attacked when castling; castle after the king moved
You cannot castle out of check, through check, or into check. You may castle
if the rook is attacked, and in queenside castling the rook may pass over an
attacked square (b1 or b8); only the king's path matters. If the king has
moved at all, even back to its square, castling is gone for good on both
sides; if one rook has moved, castling is gone on that side only. Losing the
right to castle temporarily, because of a check or a piece in the way, does
not lose it permanently.

## En passant
Also: en passant; e.p.; in passing; how does en passant work; capture en passant
French for "in passing". When a pawn moves two squares from its starting
square and lands beside an enemy pawn on the same rank, the enemy pawn may
capture it as if it had moved only one square: it moves diagonally onto the
square the pawn passed over, and the pawn is removed. This is only allowed on
the very next move; otherwise the right is gone. The capturing pawn must be
on its fifth rank (rank 5 for White, rank 4 for Black).

## Why en passant exists
Also: history of en passant; why is en passant a rule
When the two-square first pawn move was introduced in Europe in the 1400s,
pawns could slip past enemy pawns that would have been able to capture them
under the old one-square rule. En passant restores that capture. It is one
of the last rules to be standardised; Italy only adopted it fully in 1880.

## Check
Also: check; in check; what is check; how to get out of check
A king is in check when an enemy piece attacks it. The player in check must
get out of it on the next move, in one of three ways: move the king to a safe
square, capture the checking piece, or block the check by putting a piece in
between (impossible against a knight or a pawn, or a double check). A move
that leaves your own king in check is illegal. Saying "check" aloud is
polite but not required.

## Checkmate
Also: checkmate; mate; what is checkmate; shah mat
Checkmate is a check with no legal escape: the king cannot move to safety,
the checking piece cannot be captured, and the check cannot be blocked. The
game ends at once and the side that delivered mate wins. The word comes from
the Persian "shah mat", usually translated "the king is helpless" or "the king
is defeated".

## Stalemate
Also: stalemate; what is stalemate; no legal moves
Stalemate happens when the player to move has no legal moves and is not in
check. The game is a draw immediately. It is a common way for a losing player
to escape, especially against a queen in the endgame, so the winning side
must always leave the opponent a legal move until delivering mate. In some
older rules stalemate counted as a win; not in modern chess.

## Ways a game can be drawn
Also: draw; draws; how can a game be drawn; types of draws
A game is drawn by stalemate, by agreement between the players, by threefold
repetition (claimed), by the fifty-move rule (claimed), automatically by
fivefold repetition or the seventy-five-move rule, by a dead position where
neither side can ever checkmate (for example insufficient material), or when a
player runs out of time but the opponent cannot possibly checkmate.

## Threefold repetition
Also: threefold repetition; repetition of position; draw by repetition; three times
A player can claim a draw when the same position occurs three times, with the
same player to move, the same pieces on the same squares and the same
castling and en passant rights. The repetitions do not have to be in a row.
The position need not come from the same moves; only the position counts.
In this app, ask to claim the draw and the board checks that the claim is
valid.

## Fivefold repetition
Also: fivefold repetition; five times
If the same position occurs five times, the game is drawn automatically,
without either player claiming. FIDE added this rule in 2014 so that games
cannot go on forever.

## Fifty-move rule
Also: fifty move rule; 50 move rule; fifty moves
A player can claim a draw if the last fifty moves by each side (100 half-moves)
were made without any pawn move and without any capture. It exists so a
player cannot drag out a game forever with no progress. Some won endgames,
such as some queen versus two bishops positions, need more than fifty moves,
but the rule applies anyway.

## Seventy-five-move rule
Also: seventy five move rule; 75 move rule
If seventy-five moves by each side (150 half-moves) pass without a pawn move
or a capture, the game is drawn automatically, unless the last move was
checkmate. Introduced by FIDE in 2014 together with fivefold repetition.

## Insufficient material
Also: insufficient material; dead position; not enough material to checkmate
The game is drawn at once when neither side can checkmate by any sequence of
legal moves. The standard cases: king against king; king and bishop against
king; king and knight against king; and king and bishop against king and
bishop with both bishops on squares of the same colour. King and two knights
against king is not automatically drawn, since a mate position exists, though
it cannot be forced.

## Draw by agreement
Also: offer a draw; draw offer; agreed draw; how to offer a draw
Players can agree a draw at any time. Over the board, a draw is offered after
making your move and before pressing the clock; the opponent can accept, or
decline by saying so or by moving. Some tournaments ban draw offers before a
certain move (Sofia rules). In this app, a draw offer is accepted or declined
on the merits of the position.

## Resignation
Also: resign; resigning; when to resign; how to resign
A player can resign at any time, conceding the game. Strong players resign
when the position is hopeless; at lower levels it pays to play on, because
opponents often make mistakes. Resigning on your own move is common
practice, and tipping over the king is the traditional gesture.

## Time controls
Also: time control; classical; rapid; blitz; bullet; how long is a chess game
FIDE classifies games by time: blitz is 10 minutes or less per player, rapid
is more than 10 and less than 60, and standard (classical) is 60 minutes or
more, counting the increment over 60 moves. Bullet, under 3 minutes, is an
online favourite. Classical games can take several hours; a typical modern
time control is 90 minutes plus a 30-second increment per move.

## The chess clock
Also: chess clock; flag; flag fall; increment; delay; Fischer clock
Each player has their own time; after moving, a player presses the clock to
start the opponent's. Running out of time loses, unless the opponent cannot
checkmate by any legal series of moves, in which case it is a draw. An
increment (invented by Fischer) adds time after every move; a delay waits a
few seconds before the clock starts counting down. The old mechanical clock
had a little flag that fell when time ran out, hence "flagging".

## Touch-move
Also: touch move; touched piece; j'adoube; I adjust
In over-the-board play, if you deliberately touch one of your pieces you must
move it if it has a legal move, and if you touch an opponent's piece you must
capture it if you can. To straighten a piece without this obligation, say
"j'adoube" (French for "I adjust") first. Once you let go of a piece on a
legal square, the move is made. Touch-move does not apply online.

## Illegal moves in tournaments
Also: illegal move; illegal move penalty; what happens if you make an illegal move
Under the FIDE Laws, when an illegal move is completed (the clock pressed),
the position before it is restored. The first time, the opponent gets two
extra minutes (one in blitz); the second illegal move by the same player
normally loses the game. Pressing the clock without moving, or using two
hands for one move, also counts as illegal. Online and in this app, illegal
moves simply cannot be played.

## Who moves first
Also: who goes first; white moves first; why does white move first
White always moves first. This became the standard in the late 1800s. The
first move gives a small advantage: White scores slightly better than Black
in master games, roughly 55 percent of the points.

## Algebraic notation
Also: chess notation; how to read chess moves; algebraic notation; SAN; reading moves
Each square is named by its file (a to h) and rank (1 to 8), seen from
White's side: e4 is the e-file, fourth rank. A move is written with the
piece's letter, K for king, Q queen, R rook, B bishop, N knight and nothing
for a pawn, then the destination square: Nf3, e4. Captures add an x (Bxe5,
exd5), check adds + and mate #. O-O is kingside castling, O-O-O queenside,
and e8=Q a promotion. Annotations mark quality: ! good, !! brilliant, ? mistake,
?? blunder, !? interesting, ?! dubious.

## PGN and FEN
Also: PGN; FEN; portable game notation; Forsyth-Edwards notation; export the game
PGN (Portable Game Notation) is the standard text format for whole games: tag
lines such as Event, White, Black and Result, then the moves in algebraic
notation. FEN (Forsyth-Edwards Notation) describes a single position in one
line: the pieces rank by rank, whose move it is, castling rights, the en
passant square and the move counters. This app can export the game as PGN.

## Chess ratings
Also: Elo; rating; Elo rating; what is a good rating; rating system
The Elo system, created by physicist Arpad Elo and adopted by FIDE in 1970,
estimates playing strength from results: beating a stronger player gains
more points than beating a weaker one. A 200-point difference means the
stronger player is expected to score about 75 percent. Rough guide: beginners
are below 1000, club players around 1400 to 1800, masters above 2200, and
grandmasters around 2500 and up. Online sites use their own scales, often
Glicko, so numbers differ between sites.

## Chess titles
Also: grandmaster; GM; international master; IM; FIDE master; FM; titles
FIDE awards titles for life. Grandmaster (GM), the highest, needs a rating of
2500 and three grandmaster norms (strong performances in qualifying
tournaments). Below it are International Master (IM, 2400 and norms), FIDE
Master (FM, 2300) and Candidate Master (CM, 2200). There are separate women's
titles: WGM, WIM, WFM and WCM, though women can and do earn the open titles.
There are about two thousand grandmasters in the world.

## Chess960
Also: Fischer random; Chess960; freestyle chess; 960
A variant proposed by Bobby Fischer in 1996: the back-rank pieces are
shuffled into one of 960 starting positions, with the bishops on opposite
colours and the king somewhere between the two rooks. Black's pieces mirror
White's. Castling still ends with the king and rook on their usual squares.
It removes opening memorisation and rewards understanding; top players now
play it in "Freestyle Chess" events.

## Tournament formats
Also: Swiss system; round robin; knockout; how do chess tournaments work
In a round robin, every player meets every other player. In a Swiss system,
used for large open events, players with similar scores are paired each
round, so nobody is eliminated and the field sorts itself out over a fixed
number of rounds. Knockout events use short matches with faster tiebreaks.
A win scores one point, a draw half a point each, a loss nothing.

## Can a king take a king
Also: king capture; can kings touch; kings next to each other
No. A king can never move next to the enemy king, since it would be moving
into check, so kings are always at least one square apart. In real chess a
king is never captured: the game ends at checkmate, before any capture of the
king could happen.

## Can a pawn capture backwards
Also: pawn capture backwards; pawn capture sideways; pawn capture straight
No. A pawn only captures one square diagonally forward. It cannot capture
straight ahead, sideways or backward, and it cannot move straight forward
onto an occupied square.
