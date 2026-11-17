# GTA VI Twitch CCU Forecaster

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/UpTo1M/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/UpTo1M/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for
hours 2–24 of release day (2026-11-19 UTC), as five quantiles per hour, from
hour 1's print plus the launch curves of past big games carried in the case.
Entry for the [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
bounty on make.fun.

## Model

Real launch-day viewership does not stay flat: it dips overnight, then rises
as regions unlock (US Eastern 05:00Z, US Pacific 08:00Z) and peaks in US prime
time. The growth from hour 1 depends strongly on how big hour 1 is — across
37 past launches, corr(log hour-1, log viewers_19 / viewers_1) = −0.93. A
launch whose hour 1 is already huge grows only modestly; one whose hour 1 is
tiny explodes later. So the forecast is conditioned on hour-1 size:

1. From the case's `comparables`, take each past launch's 24 hourly viewers
   on its release day and its log-ratio curve `r_h = ln(viewers_h / viewers_1)`.
2. Keep the launches whose log hour-1 is within 1.75 of the case's log hour-1
   (nearest neighbours in scale); widen the window until at least 8 launches
   are kept. The 1.75 width is tuned by leave-one-out.
3. Per hour, take the median of `r_h` over the kept launches and the empirical
   quantiles of `r_h`. The five forecast quantiles blend the empirical
   quantiles 15% toward the lognormal quantiles (median + z·σ, σ from the
   inter-quartile range).
4. `q_τ(h) = p1 · exp(blended log-ratio)`, `p1 = max(1, hour-1 viewers)`.

If the case carries too few usable comparables, a hardcoded shape (the median
log-ratio curve of the big launches) and a baseline-style spread are used
instead.

## Validation

Leave-one-out over the 37 comparable launches in the case, running this exact
program on synthetic one-held-out cases:

| set | this model | persistence baseline |
|---|---|---|
| hour-1 ≥ 20,000 (n=27) | **0.247** mean pinball | 0.321 |
| all 37 launches | **0.416** mean pinball | 0.930 |

The model beats the baseline on 22/27 big-hour-1 launches and 32/37 overall.
The house baseline's own practice loss is 0.387.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Deterministic, <1s per case.

## License

MIT — see [LICENSE](LICENSE).
