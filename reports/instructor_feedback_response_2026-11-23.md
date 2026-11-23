# Instructor feedback, 23 November 2026, and what we do about it

The evaluation made five technical observations and six recommendations. This note takes each
one, says where the project stands on it today, and names the action, the owner and the week.
Items marked done are already on main. Items marked open go into the plan.

## The five observations

**1. Multimodal feasibility is not demonstrated.** Agreed, and this is now the design. The video
and audio corpora share no animals, pens, sessions or timestamps, so there is no learned fusion.
The two branches are built and scored separately. They meet only in the event record, one per pen
per five minutes, and rules on that record combine them at decision level. The report and the
README say so in those words. Action: keep the wording "decision-level rules on independent
outputs" everywhere, and never call it audio-visual fusion. Done.

**2. Dataset scope limits generalisation.** Agreed. Video is one pen, eight pigs, seven days.
Audio is six recording teams and 43 sessions. The sampling unit for video is the recording day
(we hold out whole days), for audio the session or the recording team (we hold out whole teams).
Both branches already show the drop that comes with a new unit: video 0.32 macro-F1 across days
with a spread of 0.06, audio from 0.74 within seen teams to 0.52 on a held-out team. Action: a
"sampling units and expected shifts" paragraph in the paper's data section, and a limitations
paragraph that names farms, animals, devices and environments as untested shifts. Open, paper,
week 14.

**3. Clarify data accounting.** This is the most concrete point and it is answered in the README
section "Data accounting". In short:

| Count | What it is |
| --- | --- |
| 172,800 | label rows at full rate: 12 clips x 8 pigs x 1,800 video frames |
| 12,646 | rows where a human placed the box and label (every third frame, when something changed) |
| 43,232 | rows filled in by interpolating between two human boxes |
| 116,922 | rows after a pig's last human box, carried forward |
| 57,600 | rows on the 2-per-second grid: 12 x 8 x 600 |
| 39,180 | filled-in rows kept by the gate |
| 120,974 | filled-in rows rejected by the gate |
| 6,887 | Soundwel calls |
| 6,150 | calls with all ten supplied acoustic features; the other 737, all from one team, lack them |

The 12,794 the evaluation quotes is not a number in our current tables. Every count in the paper
will carry its unit (rows, frames, calls). The 16-to-7 label mapping is now a table in the README
with counts and the reason for each decision. Audio valence is a label derived from the production
context (isolation is negative, reunion is positive, and so on), and the paper will say so and
report per-class precision and recall rather than call valence a welfare diagnosis. Done in the
README, open in the paper.

**4. The edge claim is an objective, not a result.** Agreed. What exists today: the detector is
2.6 million parameters and 5.4 MB, runs at 13.5 ms per frame on a laptop GPU and 58 ms on its CPU,
and the event record is 24 bytes. What does not exist: any measurement on the target device.
Action: a Raspberry Pi 5 is being bought this week. Week 14 measures model size, latency at three
input sizes, memory, power and temperature, in 32-bit and 8-bit, plus the accuracy after 8-bit
conversion on the held-out days. "Lightweight" will then be defined as the measured numbers.
Open, systems track, week 14.

**5. The evaluation plan is incomplete.** Mostly answered already, with one gap. Baselines
(majority class, no-model budget copy, logistic regression on supplied audio features), metrics
(macro-F1, budget error in points, AUC, mAP50), validation protocol (leave-one-day-out and
leave-one-team-out), class imbalance (balanced sample weights, macro-F1) and ablations (frame
rate, human versus predicted boxes, crop model variants) exist. The gap is per-class reporting and
confusion matrices in the write-ups, and event-level measures. Per-class precision, recall and F1
for the box model are now in the README. Event-level measures (detection rate, lag, false alarms
per day) come with the injected-anomaly evaluation. Open, week 14.

## The six recommendations

1. **Decide on paired data.** No paired data can be collected before December, so the answer is
   the one the evaluation suggests: separate models, decision-level combination, labelled as such.
   Done.
2. **Separate baselines first.** Done for both branches, see the README tables.
3. **Clarify the fusion strategy.** Decision-level rules on the event record, tested with injected
   anomalies. The rule set is drafted and is the first item of week 13. Open.
4. **Class-wise precision, recall and F1, especially rare welfare behaviours.** Box model table
   below and in the README. Rare welfare behaviours, fighting above all, are not in the seven
   scored classes. Fighting has 1,367 rows and we will test whether it can be flagged as an event
   from motion and overlap, rather than scored as a budget class. Open, week 13.
5. **Assess generalisation under lighting, occlusion, noise.** Two cheap tests: the detector and
   box model on frames with brightness changes, blur and added noise, and the audio CNN on calls
   with added noise and gain changes. Both are perturbation tests on the held-out days, not new
   data, and the paper will say so. Open, week 14 if time allows, otherwise stated as a limitation.
6. **Demonstrate edge feasibility, including offline store-and-forward.** Pi measurements as in
   observation 4, plus a store-and-forward test under emulated 2G with the link taken down and
   restored. Open, week 13 and 14.

## Per-class results for the box model, six-day protocol

Human-labelled rows on the 2-per-second grid, 2 December excluded, every other day held out in
turn. Support is the number of human-labelled rows of that class.

| Behaviour | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| walk | 0.55 | 0.67 | 0.61 | 2,464 |
| investigating | 0.53 | 0.41 | 0.46 | 3,638 |
| drink | 0.32 | 0.45 | 0.37 | 163 |
| eat | 0.40 | 0.26 | 0.32 | 467 |
| standing | 0.26 | 0.30 | 0.28 | 1,672 |
| sleep | 0.23 | 0.30 | 0.26 | 498 |
| lying | 0.17 | 0.17 | 0.17 | 584 |

The confusion matrix is in `reports/v2_sixday_confusion_2026-11-23.csv`. The three still postures
(sleep, lying, standing) are confused with each other and with investigating, which is what a box
that does not move looks like. Walking is the strongest class. Drinking is small but found, because
it happens in one place in the pen.

## Key message

The evaluation's closing line is that the contribution must be reliable, resource-efficient
welfare monitoring under realistic conditions, not a combination of two models. Our reading of the
same evidence is that the honest contribution is three measured things: how much accuracy
survives a new day or a new recording team, how little data has to leave the farm, and what the
monitor costs to run on a Raspberry Pi. The fusion rules sit on top of those as a clearly labelled
design, evaluated with injected anomalies because real ones do not exist in the data.
