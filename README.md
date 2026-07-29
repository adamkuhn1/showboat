# showboat

2D bar-pool game with a real trained ML opponent. Owned by the `build-showboat`
agent in Phase 2.

**v1 (see `PLAN.md` §5):** standard bar-pool rules, full games, 2D. AI opponent is a
**real trained model** biased toward bank shots and multi-wall combos (**no jump, no
massé**), wins convincingly, with a **live reasoning overlay** driven by real
decision data. Short "thinking" pause acceptable.

**Hard constraint:** real trained ML, not heuristics dressed up. Trick shots must be
emergent. Overlay must reflect actual decisions. Jump/massé impossible at the
action-space level.

Not yet initialized. Follow `/research/showboat-ml.md` + `/research/synthesis.md`.
