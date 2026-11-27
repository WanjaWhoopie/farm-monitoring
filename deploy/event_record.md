# The event record

One record per pen per five-minute window. This is the only thing that leaves the farm. The
device computes it from the detector's tracks (vision fields) and from the call detector (audio
fields), applies the rules in `rules.json` to it, and sends it. A web dashboard receives these
records and nothing else, so everything it shows has to come from these fields.

Defined in `scripts/event_summary.py`. Worked examples in
`reports/deployability_2026-11-19/summaries_all_clips.jsonl` (real vision fields from the
Edinburgh clips, audio fields empty because those pens had no microphone).

## JSON form

```json
{
  "v": 1,
  "pen": 1,
  "t": 1575288000,
  "win_min": 5,
  "pigs_seen": 7.6,
  "budget": {"sleep": 0.03, "investigating": 0.38, "eat": 0.02, "standing": 0.08,
             "lying": 0.20, "walk": 0.25, "drink": 0.04},
  "motion_px_s": 41.2,
  "audio": {"calls": 17, "neg_valence": 0.18},
  "flags": [],
  "severity": 0,
  "src": "pen monitor 1"
}
```

| Field | Meaning | From |
| --- | --- | --- |
| `v` | record version | fixed |
| `pen` | pen number on the farm | config |
| `t` | window start, Unix seconds | clock |
| `win_min` | window length in minutes (5, 15 or 60) | config |
| `pigs_seen` | mean number of boxes per frame over the window | detector |
| `budget` | share of pig-time in each of the seven behaviours, sums to 1 | behaviour model on the tracks |
| `motion_px_s` | mean box speed in pixels per second | tracks |
| `audio.calls` | number of calls detected in the window | call detector |
| `audio.neg_valence` | share of those calls the valence model scored negative, null when `calls` is 0 | audio model |
| `flags` | names of the rules that fired this window, from `rules.json` | rules |
| `severity` | 0 normal, 1 watch, 2 alert, 3 urgent (see `rules.json`) | rules |
| `src` | free text, where the record came from | config |

Compact JSON is about 290 bytes. The dashboard should accept the fields it knows and ignore
any it does not, so the device can add fields without breaking it.

## Binary form, 24 bytes, little-endian

For SMS fallback. Five of these fit in one 140-byte SMS.

| Bytes | Field | Encoding |
| --- | --- | --- |
| 0 | version | u8 |
| 1 | pen | u8 |
| 2 to 5 | window start | u32, Unix seconds |
| 6 | window length | u8, minutes |
| 7 | pigs seen | u8, value x 10 |
| 8 to 14 | seven behaviour shares | u8 each, 0 to 255 = 0 to 100 %, order sleep, investigating, eat, standing, lying, walk, drink |
| 15 to 16 | motion | u16, pixels per second x 10 |
| 17 | calls | u8 |
| 18 | negative-valence share | u8, 0 to 255 = 0 to 100 % |
| 19 | flags | u8, one bit per rule: bit 0 movement, 1 rest, 2 call, 3 combined, 4 count, 5 feeding, bits 6 to 7 severity |
| 20 | CRC-8 | polynomial 0x07 over bytes 0 to 19 |
| 21 to 23 | reserved | zero |

Encoder, decoder and CRC in `scripts/event_summary.py` (`encode_binary`, `decode_binary`,
`crc8`). The struct format is `<BBIBB7BHBBB` followed by the CRC byte and three zero bytes.
`deploy/check_handover.py` round-trips a record with flags and severity and checks that a
corrupted byte is rejected.

## What the dashboard should show

Per pen: the seven shares over the day as a stacked area, `pigs_seen` and `motion_px_s` as
lines, `audio.calls` as bars with `neg_valence` as a line, and the severity as a coloured band
along the time axis. A list of windows with severity 2 or 3, newest first, is the alert feed.
The baseline band the rules compare against (median and spread per hour of day) is learned on
the device from its first seven days and can be sent once a day as a separate small record if the
dashboard wants to draw it.
