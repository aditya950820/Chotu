# Chotu Flight School Cheat Sheet

Copy of the shared Claude doc "Chotu Flight School Cheat Sheet" (5 Nov 2026), saved here so it lives in the repo.
The doc stays the place to edit; refresh this copy when it changes.

## At a glance

Chotu learns by guessing the next token, getting told how surprised it was, and nudging its knobs. Everything in the eight lessons is about watching that loop and spending money on it wisely.

**One training step, the loop Chotu repeats 19,181 times:**

1. Pick a fresh pile of 256 papers.
2. Chotu guesses every next token and notes its surprise.
3. Each paper gives advice: which way to turn each knob.
4. Blend the 256 pieces of advice into one direction.
5. Turn every knob a little that way. The size of the turn is the learning rate.
6. Log the average surprise as one dot on the loss curve.
7. Every so often: sit the exam (no knob changes) and save a checkpoint.

**Numbers to remember:**

| What | Number |
| --- | --- |
| Chotu's knobs | 180,925,056 |
| Tiny twin's knobs | 7,079,552 |
| Cards in the deck (tokens) | 49,152 |
| Total reading | about 10 billion tokens |
| Steps | 19,181 |
| Papers per step (batch) | 256 |
| Warm-up | first ~700 steps |
| Slow-down | last 20% (3,836 steps) |
| LR test | 1.5e-3 vs 3e-3, 0.5B tokens each |
| H100 rent | ₹330 an hour |
| Blind-guessing start | 10.8 (untrained Chotu measured 10.91) |
| Common words only | 7.53 |
| SmolLM2-135M (rough finish line) | 2.77 |
| Kill rule | loss below about 6 by step 300 |

## Lesson 1: The guessing game

**Loss = Chotu's average surprise at the next token.** Lower is better.

**Analogy:** Chotu has a deck of 49,152 cards (tokens) and 100 chips. Before each card is turned, it spreads chips across the deck. Lots of chips on the right card = barely surprised. Almost none = very surprised.

- Untrained Chotu spreads chips evenly, which gives a surprise of **10.8**. A run that starts far from 10.8 is set up wrong.
- Just knowing which tokens are common ("the", ",", ".") cuts surprise a lot, down to **7.53**, without reading the sentence at all.
- Reading the sentence is what gets you further: SmolLM2 reaches **2.77**.
- **Tokens are word pieces, not words.** "Kevlar" is two cards: "K" + "evlar". SmolLM2 put 99% on "evlar" after "K" because it read the passage about the fibre.

**Remember:** below about 6 by step 300 means Chotu is already reading the sentence, not just betting on frequent words.

## Lesson 2: The shape of learning

**A healthy loss curve has three phases.** Think of a child learning to read. (imagine that graph)

| Phase | When | What it looks like | What's being learned |
| --- | --- | --- | --- |
| A · The cliff | First few hundred steps | Loss drops fast from 10.8 | Which words are common |
| B · The long slope | Most of the run (15+ hours) | Nearly flat, small wiggles | Grammar, then facts and meaning |
| C · The slowdown | Last 20% | One more noticeable drop | Strides shrink, Chotu settles |

- **The boring slope is where the valuable learning happens.** The cliff is easy wins.
- **Wiggles in phase B are normal.** Each step reads different papers, some harder than others.

**Sick shapes:**

| You see | It means |
| --- | --- |
| Flat from the start | Nothing is learning. A bug |
| Sudden jump that never comes back | Loss spike. Alert |
| NaN | Numbers blew up. Stop at once |
| Practice improves, exam gets worse | Memorising. Unlikely when every paper is read once |

## Lesson 3: Nudge size (learning rate)

**Learning rate = stride = nudge size = step size.** It's how boldly Chotu follows the advice each step. ("A step" on its own means one round of the loop; "step size" means the learning rate.)

**Analogy:** walking down a foggy valley to the lowest point. Chotu only feels the slope under its feet.

| Stride | What happens |
| --- | --- |
| Too small | Safe, but burns ₹330/hour going nowhere |
| Just right | Biggest stride that still settles |
| Too big | Bounces wall to wall, wobbles or spikes |
| Far too big | Flies out of the valley, NaN |

**Reading the notation:** "e-3" means move the decimal 3 places left.

| Written | Plain number |
| --- | --- |
| 3e-4 | 0.0003 |
| 1e-3 | 0.001 |
| 3e-3 | 0.003 |
| 1e-2 | 0.01 |
| 3e-2 | 0.03 |
| 1e-1 | 0.1 |

- **Warm-up (first ~700 steps):** at step 0 the advice is mostly noise, so start with baby steps.
- **Slow-down (last 20%):** near the floor, long strides hop side to side. Shrinking them lets Chotu settle.
- **Early warning before NaN:** wobbling loss, gradient often hitting the clip, exam score getting worse mid-run.
- **Nudge Size Lab result:** 3e-3 won on the tiny twin; 1e-1 never crashed but finished worst. A too-big stride can fail quietly, which is why there's an LR test.

**Not one-size-fits-all:**

- Bigger models need smaller strides (GPT-3 125M used 6e-4; 175B used 6e-5).
- Fine-tuning uses 10 to 100 times smaller strides than pretraining (around 1e-5 to 2e-5; LoRA often around 1e-4 to 2e-4).
- Changing the batch changes the best stride.
- How to choose: start from a published recipe of similar size, test two or three strides, the lowest exam score wins.

## Lesson 4: The batch

**Batch = how many papers Chotu reads before it's allowed to change anything.** Chotu's plan uses 256.

**Analogy:** a cricket coach gives one instruction after watching Chotu bat.

| Coach watches | Advice quality | Instructions per season |
| --- | --- | --- |
| 1 ball | Wild, swayed by a freak delivery | Very many |
| An over (like 64) | Mostly right, sometimes tilted | Many |
| A session (like 256) | Steady, shows real weaknesses | Fewer |
| Whole season | Perfect | About three |

- **Total reading is fixed.** The batch only decides how you slice it: many shaky instructions or fewer reliable ones.
- **Extra papers help less and less.** Past some point you pay for more watching with no better advice.
- **Big piles matter more late in training.** Early flaws are obvious; late flaws are subtle and need more papers to spot.
- **Batch and stride are linked.** Shaky advice needs a smaller stride. Change the batch and the LR test needs redoing.
- **Money:** the H100 can handle a big pile as easily as a small one. A small batch leaves it partly idle.

## Lesson 5: Practice paper vs final exam

**Training loss is the practice paper; validation loss is the board exam on locked-away papers Chotu never learns from.**

**Exam workflow:** pause training, give the same fixed exam papers, record surprise, **turn no knobs**, log one dot, resume.

**Analogy:** Student A memorises the practice paper and fails the board exam. Student B understands and scores the same in both. Only the unseen exam tells them apart.

**The twist:** every pile is fresh and each paper is read once, so Chotu can't really memorise. On a healthy run **the two lines travel close together.**

| You see | It usually means |
| --- | --- |
| Both fall together | Healthy |
| Practice falls, exam rises | Memorising; suspect repeated data |
| Exam sits a steady gap above | Exam text differs from the training mix |
| Exam better than practice | Suspicious: exam papers leaked into training, or easier text |
| Exam worsens mid-run, then recovers | Stride too big |

**Bits-per-byte:** surprise per letter instead of per token. Different tokenizers make loss unfair to compare (marks out of 50 vs out of 100). Needed for Chotu vs GPT-2 or Llama; not needed for Chotu vs SmolLM2 (same tokenizer).

## Lesson 6: Speed and money

**Speed is money.** The rent (₹330/hour) and the reading (about 10B tokens) are fixed, so reading speed decides the bill.

**Analogy:** a rented factory with 1,000 workers. Tokens per second = boxes out the door. MFU = share of workers actually building boxes.

| Idle workers in the factory | In Chotu's run |
| --- | --- |
| Raw material trucks too slow | Data loader can't feed the GPU |
| Orders too small | Batch too small |
| Waiting for the manager to sign forms | Too much CPU–GPU back-and-forth |
| Stopping to count inventory | Exams and checkpoint saves |

- **GPU utilisation near 100% doesn't mean useful work.** Lights on, everyone clocked in, some carrying the same box back and forth. **MFU is the honest number.** Small models like Chotu usually land well below big-run levels.

**First 10 minutes:**

1. **0–2 min:** no errors, GPU memory well used, first loss near 10.8.
2. **2–5 min:** ignore the slow first steps; then tokens/sec should hold steady. Compare with the plan.
3. **5 min:** read projected hours and ₹ cost.
4. **By step 300:** loss below about 6.
5. **Decide:** keep, fix and restart, or kill.

**Kill early:** stopping at 10 minutes costs about ₹55; finding a problem at hour 12 costs about ₹4,000. Ignore money already spent (sunk cost); only ask whether the next hour is worth ₹330. Turn off idle instances.

## Lesson 7: Crashes and checkpoints

**A checkpoint is a save point.** A crash only costs the time since the last save, instead of the whole run.

**Analogy:** saving a video game. A good save keeps your character, inventory, quest progress and map position, not just your level.

| A full checkpoint saves | If it's missing |
| --- | --- |
| The knobs | Loss jumps back near 10.8 |
| The coach's notebook (optimizer memory) | Spike after resume, slow recovery |
| Step number and stride schedule | Warm-up restarts, or slow-down starts early |
| Which papers were read | Re-reads seen papers or skips unread ones |
| Random settings | Small drift from the original run |

- **Save every 30–60 minutes** for a run like this: less risk than rare saves, fewer pauses than constant ones.
- **Keep the last few saves,** so you can step back past a loss spike.
- **Save to storage that survives the machine.**
- **Write to a temporary file, then swap it in,** so a crash mid-save can't destroy the good copy.

**Machine dies at hour 12:** don't restart from scratch → pick the latest good checkpoint → load everything on a fresh machine → watch the first minutes (loss should continue where it was, stride should not show warm-up again) → log it.

**Fire drill before run day:** run the tiny twin 100 steps straight; run again, kill at 60, resume to 100. **The two lines must match exactly.**

## Lesson 8: What the knobs learned

**Loss says Chotu guesses better on average, not what it learned.** Same marks can hide very different students.

**Analogy:** a parent-teacher meeting. Talk to the child (behaviour tests) and get an X-ray (look inside the knobs).

**Talking to Chotu:**

- **Writing samples** from the same five prompts at each checkpoint: soup → common words → grammar → on-topic sentences → fluent paragraphs.
- **Fill-in-the-blank:** where it bets on "The capital of France is ___". Compare with SmolLM2.
- **Copying test:** show a random sequence twice. A trained model's surprise drops sharply the second time. The clearest sign it learned to use the context.

**X-ray:**

- **Word map:** related words should end up as neighbours (Monday next to Tuesday).
- **Where it looks:** attention habits form (previous word, matching quotes, copying).
- **Knob health:** no layer exploded or went dead compared with step 0.

**Workflow:** keep checkpoints from across the run (step 0, 300, ~1,000, ~5,000, mid, end) → run the same report card on each → line them up → compare the final one with SmolLM2 and GPT-2 small (bits-per-byte when tokenizers differ).

**Traps:** cherry-picking one good sample; expecting reliable facts from a 180M model; comparing benchmark scores against big models.

**Answers to the Lesson 8 questions:**

1. Fluent but says Sydney is Australia's capital → not a failure. Small models learn form before facts, and Sydney appears near "Australia" more often.
2. Same loss, one passes the copying test → keep the copier. That skill underlies using prompts and context.
3. Mid-run checkpoints → they show the trend, when skills appeared, when something broke, and give you a fallback.

## Training checklist

Tick these in order, from the free work on your Mac to the report card after the run.

**Phase 1: Before renting the GPU (free, on your Mac)**

- [ ] Training shards tokenized; exam shard kept separate and never used for training
- [ ] Untrained Chotu scores near 10.8 on your data (flight simulator check)
- [ ] Tiny twin trains for a few hundred steps and its loss falls (smoke test)
- [ ] Checkpoint fire drill passed: stop-and-resume line matches the straight run
- [ ] Config confirmed: batch 256, 19,181 steps, warm-up ~700 steps, slow-down over the last 20%
- [ ] Exam interval and checkpoint interval set (checkpoints every 30–60 min)
- [ ] Checkpoints write to storage outside the GPU machine, via a temp file then swap
- [ ] Dashboard logs: loss, exam loss, gradient size, learning rate, tokens/sec, projected hours and ₹
- [ ] Budget set: maximum ₹ for the run and the point where you'd kill it
- [ ] Report card ready: five prompts, fill-in-the-blanks, copying test, word-map words
- [ ] No idle GPU instances running

**Phase 2: LR test (on the H100)**

- [ ] Run 1.5e-3 and 3e-3 on 0.5B tokens each, same data, same everything else
- [ ] Compare final exam scores; the lower one wins
- [ ] Note the result and why in the run log

**Phase 3: Launch, the first 10 minutes**

- [ ] 0–2 min: no errors, GPU memory well used, first loss near 10.8
- [ ] 2–5 min: ignore the slow first steps, then tokens/sec holds steady and matches the plan
- [ ] 5 min: projected hours and ₹ are within budget
- [ ] Stride chart shows warm-up climbing as planned
- [ ] Step 300: loss below about 6
- [ ] First checkpoint appears in storage and loads
- [ ] Decision made: keep, fix and restart, or kill

**Phase 4: During the run (glance every hour or two)**

- [ ] Loss still trending down; long flat stretches with small wiggles are fine
- [ ] Exam dots tracking close to the practice line
- [ ] No spikes, no NaN, gradient not constantly hitting the clip
- [ ] Tokens/sec steady (short regular dips are exams and saves)
- [ ] New checkpoints arriving; last few kept
- [ ] Cost on track with the projection
- [ ] At the last 20%: stride chart starts shrinking, loss takes one more drop

**Phase 5: If the machine dies**

- [ ] Don't restart from scratch
- [ ] Pick the latest good checkpoint (or an earlier one if a spike came first)
- [ ] Load everything on a fresh machine: knobs, notebook, step, schedule, data position
- [ ] First minutes: loss continues where it left off, no warm-up restart, speed as before
- [ ] Log what happened, which checkpoint, time lost

**Phase 6: At the end**

- [ ] Final checkpoint saved and copied to safe storage
- [ ] Final exam score and bits-per-byte recorded
- [ ] GPU shut down straight away

**Phase 7: Report card (after, off the GPU)**

- [ ] Same prompts, blanks and copying test run on checkpoints from across the run
- [ ] Writing improves from soup to fluent sentences
- [ ] Copying test: big drop in surprise on the second copy
- [ ] Word map: related words are neighbours
- [ ] Knob health: no exploded or dead layers
- [ ] Compared with SmolLM2 and GPT-2 small
- [ ] Write-up saved to the repo

**Warning signals and what to do:**

| Signal | Lesson | Action |
| --- | --- | --- |
| First loss far from 10.8 | 1 | Stop. Setup is wrong |
| Loss not below about 6 by step 300 | 2 | Stop. Something is broken |
| Speed far below plan, cost way over budget | 6 | Stop, fix the bottleneck, restart |
| NaN | 2 | Stop at once |
| Loss spike | 2 | Alert, watch closely |
| Gradient often hitting the clip, wobbly loss | 3 | Stride likely too big |
| Exam rises while practice falls | 5 | Check for repeated or leaked data |
| Exam better than practice | 5 | Check for leakage |
| Speed slowly falling over hours | 6 | Something leaking or clogging |
| Short regular speed dips | 6 | Normal: exam or checkpoint |
| After resume: loss near 10.8 | 7 | Knobs didn't load |
| After resume: spike then slow recovery | 7 | Notebook missing or warm-up restarted |
| After resume: loss drops suspiciously fast | 7 | Re-reading seen papers |

## Terms to remember

| Term | Plain word | What it means | Lesson |
| --- | --- | --- | --- |
| Token | A card | A word piece the text is chopped into; "Kevlar" = "K" + "evlar" | 1 |
| Tokenizer | The card cutter | The tool that chops text into tokens; different models cut differently | 1, 5 |
| Vocabulary | The deck | All the tokens a model knows; Chotu's has 49,152 | 1 |
| Next-token prediction | The guessing game | The only task in pretraining: guess the next card | 1 |
| Loss | Average surprise | How surprised the model is by the real next token, averaged; lower is better | 1 |
| Perplexity | How many cards it's torn between | Loss in another form; untrained Chotu is torn between all ~49,000 cards | 1 |
| Parameters / weights | Knobs | The numbers the model adjusts while learning; Chotu has ~181M | 1 |
| Shard | A box of papers | One chunk file of the training or exam text | 1 |
| Loss curve | The line you watch | Loss plotted over steps; healthy = cliff, long slope, slowdown | 2 |
| Loss spike | A sudden jump | Loss jumps up and may not come back; alert | 2 |
| NaN | Numbers blew up | "Not a number"; training is broken, stop at once | 2 |
| Overfitting | Memorising | Doing well on practice, worse on new text | 2, 5 |
| Gradient | The slope / the advice | Which way and how strongly each knob should turn | 3 |
| Gradient clipping | Capping extreme advice | Limits how big one step's advice can be, to avoid blow-ups | 3 |
| Learning rate | Stride, nudge size | How boldly the knobs move each step | 3 |
| LR schedule | The stride plan | How the stride changes over the run: warm up, hold, slow down | 3 |
| Warm-up | Baby steps | Small strides at the start while advice is noisy | 3 |
| Decay / cooldown | Slowing down | Shrinking strides at the end so Chotu settles | 3 |
| LR test / sweep | Trying a few strides | Short runs at different strides; lowest exam score wins | 3 |
| Pretraining | Learning from scratch | Training random knobs on lots of text; normal strides | 3 |
| Fine-tuning | Polishing a trained model | Further training on a small dataset; much smaller strides | 3 |
| LoRA | A small add-on | Fine-tuning only a small extra set of knobs | 3 |
| Step | One round | Read a pile, get advice, turn the knobs once | 4 |
| Batch | The pile | Papers read before one nudge; Chotu uses 256 | 4 |
| Critical batch size | Big enough pile | Past this size, extra papers barely improve the advice | 4 |
| Optimizer (AdamW) | The coach | The method that turns advice into knob changes | 4, 7 |
| Training loss | The practice paper | Surprise on the pile about to be learned from | 5 |
| Validation loss | The board exam | Surprise on locked-away papers that are never learned from | 5 |
| Held-out data | Locked-away papers | Text kept out of training, used only for the exam | 5 |
| Data leakage | Exam papers in the practice pile | Exam text slipped into training; exam scores look falsely good | 5 |
| Bits-per-byte | Surprise per letter | A loss that's fair to compare across different tokenizers | 5 |
| H100 | The rented factory | The GPU Chotu trains on; ₹330 an hour | 6 |
| Tokens per second | Boxes out the door | How fast Chotu reads; sets the bill | 6 |
| GPU utilisation | Lights on | Whether the GPU is running anything; can mislead | 6 |
| MFU | Workers actually working | Share of the GPU's full power doing useful model work | 6 |
| Data loader | The supply trucks | The code that feeds papers to the GPU | 6 |
| Kill rules | When to pull the plug | Pre-agreed signals that mean stop the run | 6 |
| Sunk cost | Money already gone | Spent money that shouldn't decide whether to continue | 6 |
| Checkpoint | A save point | A saved copy of everything needed to continue training | 7 |
| Optimizer state | The coach's notebook | The optimizer's memory per knob; must be in the checkpoint | 7 |
| Resume | Loading the save | Continuing a run from a checkpoint | 7 |
| Fire drill | Practice crash | Kill and resume a small run to prove checkpoints work | 7 |
| Preemptible / spot GPU | A cheaper rental that can be taken away | Only safe with reliable checkpoints | 7 |
| Embeddings | The word map | Each token's position; related words should end up close | 8 |
| Attention | Where it looks | How each word picks which earlier words to use | 8 |
| Induction / copying | Spotting a repeat | Predicting a sequence it has already seen in the text | 8 |
| Benchmark | A standard test | Shared multiple-choice tests; small models score near chance | 8 |
| SmolLM2-135M | The rough finish line | An already-trained model of similar size and same tokenizer | 1, 8 |
| Tiny twin | Chotu's 7M sibling | A small copy used for cheap experiments on CPU | 3, 7 |
