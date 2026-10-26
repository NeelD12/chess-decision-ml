# Move Quality Thresholds (Step 9b)

Based on commonly-used chess analysis conventions (Lichess community
documentation and independent chess-analysis sources), we adopt centipawn
loss (CPL) thresholds of roughly 50 / 100 / 300 for inaccuracy / mistake /
blunder, rather than arbitrary values.

| Category    | CPL Range   |
|-------------|-------------|
| Excellent   | < 10        |
| Good        | 10–49       |
| Inaccuracy  | 50–99       |
| Mistake     | 100–299     |
| Blunder     | ≥ 300       |

An earlier draft used stricter thresholds (10/25/50/100), which flagged
~17% of all moves as blunders - implausibly high compared to real player
blunder rates. That mismatch is what prompted this revision.