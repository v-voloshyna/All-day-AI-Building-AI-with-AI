# Making "Building AI with AI: what are the limits" interactive — plan (v2)

Status: plan only. Nothing here is built yet. v1 of the talk (see `PRESENTATION_v1.pdf`) is presenter-driven:
the presenter pastes prompts, the agent works, the oracle reveals. The audience watches.

## The idea in one line

Every demo already has a reveal. Interactivity means making the room **commit to a prediction before
each reveal** and letting the room **supply the inputs** the model is judged on. The audience becomes
the oracle; the presenter becomes the referee.

## Design rules

1. One interactive beat per demo, never two. Each beat costs 60–120 s; the budget comes from cutting
   the optional follow-ups (`01b_followup`, `02b_hard_tests`, `03b_fix`), not from the reveals.
2. Every beat has a zero-tech fallback (hands up, shout out) so a dead Wi-Fi does not kill the talk.
3. Predictions are written down (poll or slide) so the gap between "what we expected" and "what we
   measured" is on screen, next to the agent's own claim. Same shape as the `truth` table.
4. Audience inputs become data files the existing oracle already understands (`ev audience …`),
   so no new evaluation code is needed. Only collection is new.

## Beats, by demo (ranked by payoff / effort)

| # | Demo | Beat | Payoff | Effort | Fallback |
|---|------|------|--------|--------|----------|
| 1 | 2 (own tests) | **"Fool the model."** QR code → short form. Everyone types one review they think the model will get wrong. A tiny local service classifies each submission live and shows a running board: fooled / not fooled, with the sentence. Export to `audience.csv`, run `ev audience own_tests --show`. | Very high. The room writes the adversarial set; the agent's own tests looked thorough until a stranger typed "Loved the part where it ended." | Medium: a ~120-line Flask/FastAPI app + one HTML board, served on the venue LAN or a tunnel. | Ask 5 people to shout a sentence; type them into `predict.py --text` live. |
| 2 | 3 (holdout) | **Prediction poll.** Before `ev holdout baseline`: "The agent reported 99.7%. What will it score on reviews it has never seen?" Buckets: >95 / 85–95 / 70–85 / <70. Show the distribution, then the number. | High. The gap becomes the room's surprise, not the presenter's. | Low: Slido/Mentimeter question, or the self-hosted poll from beat 1. | Hands up per bucket. |
| 3 | 3 (leak) | **"Spot the leak."** Put one `dupes --pairs 1` pair on screen *before* explaining it: two rows, two labels, no commentary. 45 s pair discussion: "what is wrong with this dataset?" Then run `dupes`. | High. Turns the most technical reveal into a puzzle. | None. One pre-rendered slide. | — |
| 4 | 4 (loop) | **The loop as a betting game.** While `loop` runs (started 3 min earlier, output on screen): "Which iteration will be the one you would not merge?" and "Will it ever touch eval.csv?" Collect bets, narrate the lines as they land, open the `git diff` on the iteration the room picked. | High when the agent cheats; still good when it stays honest (eval up, holdout flat). | Low: poll question + the existing `loop.log` tail. | Shout-out of iteration numbers. |
| 5 | 5 (synthetic) | **Human vs. model: "which ones are synthetic?"** Five sentences on a slide: three from the agent's `synthetic.csv`, two real negatives from `holdout.csv`. Vote. Then `div` shows the repeated 4-word "tells" the room half-noticed. | Medium-high. Makes "low diversity" visceral. | Low: one slide built from the rehearsal files; vote by hands. | — |
| 6 | all | **Live scoreboard.** Second window/screen shows the `truth` table refreshing (`watch -n 5 truth` or a `--watch` flag). Three columns: agent reported / oracle on visible data / oracle on holdout. Each demo adds a row while the audience watches. | Medium. Replaces the closing table with a running narrative. | Low: `truth` already exists; `--watch` is 10 lines. | Run `truth` manually at the end as in v1. |
| 7 | close | **"Name the limit."** Everyone writes one limit in ≤7 words (poll free-text or a shared doc). Show the word cloud; presenter maps the top 3 onto the five demos. | Medium. Gives the room the last word and shows what landed. | Low. | Three volunteers say theirs aloud. |
| 8 | 1 (build) | **Pre-commitment.** Before pasting `01_build`: "How long and how much will this take?" and "What accuracy will it report?" Two quick polls. Reveal: ~40 s, ~$0.11, 99.7%. | Medium. Sets up the whole talk: fast, cheap, confident. | Low. | Hands up. |

Recommended for v2: beats 1, 2, 3, 4 and 6. Add 5, 7 and 8 only if the dry run comes in under 27 minutes.

## What has to be built (backlog, in order)

1. **`oracle/audience/server.py`** — collects submissions (`text`, optional `expected_label`), calls the
   repo's `predict()` in a subprocess so a broken `predict.py` cannot take the server down, stores
   rows in `audience.csv`, serves `/board` (auto-refreshing table: sentence, expected, predicted, ✓/✗,
   running "fooled" count). Rate-limit per IP; strip anything over 300 chars; no names collected.
   Est. 2–3 h including a test with 30 fake submissions.
2. **`oracle/audience/qr.png` + slide** — QR code to the form URL. Needs the venue URL decided (LAN IP,
   or a Cloudflare/ngrok tunnel if the venue network blocks peer traffic). Est. 30 min + venue check.
3. **Poll questions file `oracle/audience/polls.md`** — the exact wording and buckets for beats 2, 4, 7, 8
   so they can be pasted into Slido/Mentimeter in 5 minutes. Est. 30 min.
4. **`truth.py --watch`** — re-render every 5 s; optional `--big` mode (larger font via `figlet`-free
   plain spacing) for projection. Est. 30 min.
5. **Two pre-rendered slides** — the "spot the leak" pair (beat 3) and the "which are synthetic" five
   (beat 5), generated from the rehearsal files by a 20-line script so they match the live data. Est. 1 h.
6. **Revised run of show** (below) rehearsed twice with a timer. Est. 2 × 35 min.
7. **Fallback kit** — a recorded screen capture of one full run, plus the v1 PDF printed. Est. 1 h.

Total: roughly one working day, most of it beat 1.

## Revised 30-minute run of show (v2 draft)

| Clock | Segment | Interactive beat |
|-------|---------|------------------|
| 0:00 | Framing slide; paste `01_build` so the agent works during the next beat | Beat 8: two polls (time/cost, accuracy) |
| 2:00 | Demo 1: agent reports; `ev eval baseline --reported` matches | Reveal poll vs. 99.7% |
| 5:00 | Demo 2: paste `02_tests`; while pytest runs, open the QR | Beat 1: "fool the model" opens, stays open until 12:00 |
| 7:00 | `cat tests/*.py`; `ev adversarial own_tests --show` | Board on screen: first audience sentences land |
| 9:00 | Demo 3: poll, then `ev holdout baseline` | Beat 2: prediction poll |
| 11:00 | "Spot the leak" slide, pair talk, then `dupes --pairs 2` | Beat 3 |
| 13:00 | Start `loop` in the background; paste `03_why` and read the agent's explanation | Beat 4: place bets while the loop runs |
| 16:00 | Demo 4: narrate loop lines; `git diff` the iteration the room chose; `ev holdout loop` | Beat 4 reveal |
| 20:00 | Demo 5: paste `05_synth` (pre-run output as fallback); five-sentence slide; `div` | Beat 5 vote |
| 24:00 | `ev audience own_tests --show` on everything the room submitted | Beat 1 payoff |
| 26:00 | Scoreboard (`truth --watch` has been visible throughout); the five limits | Beat 6 |
| 28:00 | "Name the limit" + close | Beat 7 |
| 30:00 | End; Q&A if the slot allows | |

## What the 2026-10-20 rehearsal changes

- **Two branches, not one.** Three of four agent runs found the data leak themselves (see the PDF, section 9).
  The prediction poll in beat 2 must be worded so it works either way: "The agent reported X. What will it
  score on reviews it has never seen?" If X is already ~75%, the poll becomes "will eval.csv agree?" (it won't: 0.94).
- **Beat 1 is stronger than planned.** The agent's own suite marked negation and sarcasm as *expected failures*
  (300 passed, 24 xfailed). Show the green bar, then open the audience board: the room's sentences are the
  tests the agent chose not to fail on.
- **Beat 4 needs a second bet.** The real loop never touched eval.csv; it raised the score to 1.000 and wrote
  "memorisation" in every rationale. Add the bet "will the agent *say* the gain is fake?" next to "will it cheat?".
  Either answer makes the point: the harness reads the number, not the sentence.
- **Beat 5 has its slide already.** `synthetic.csv` from run B repeats "what more could anyone want?" 43 times
  in 300 rows; pick the five sentences from `oracle/rehearsal/run3_sonnet_found_leak/demo5/`.

## Risks and open decisions

- **Network.** Beat 1 needs attendees to reach the laptop. Decide: venue LAN (ask the organiser for an
  SSID that allows peer traffic) vs. a tunnel (depends on the venue's outbound rules). Test on site.
- **Agent latency.** Live agent runs vary from 40 s to several minutes. Every agent step must have the
  rehearsal output ready to paste. The loop should run from a plain terminal, not inside an interactive
  agent session.
- **Content moderation.** Free-text submissions go on a big screen. The board needs a presenter-side
  "hide row" control and a profanity filter; keep the form anonymous.
- **Time.** v1 is already dense. v2 drops the three optional follow-up prompts. If the room is small
  (<15 people), replace polls with hands and keep only beats 1, 3 and 6.
- **Headless auth.** During the 2026-10-20 rehearsal, `claude -p` failed with an expired OAuth session.
  Run `claude login` on the presentation machine the day before; the loop depends on it.
