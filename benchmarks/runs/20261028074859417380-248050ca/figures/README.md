# LFM3 showcase plot styling

This presentation update uses the unchanged report
`20261027T185932Z-b917d23ac845`, native run
`20261028072145009000-65166f09` and llama.cpp run
`20261027185932613950-5409e768`. No measurements were rerun or changed.

The plot follows the other README showcases: outlined DejaVu Sans text (the
fallback in the Segoe UI/DejaVu Sans theme), a left-aligned title and subtitle,
blue AKAS/circle and orange llama.cpp/square series, light horizontal grids,
muted axes, comma-formatted rates, and a shared legend. Error bars show the
actual minimum and maximum of the three retained samples; lines show medians.

The previous LFM3 chart and its original recipe remain immutable in run
`20261028073053545412-431db636`.

From this bundle's root, regenerate both SVG and PNG with:

```sh
python figures/plot.py figures/report.json figures/throughput.svg
```

The tracked equivalent is `python -m tools.lfm3_plot REPORT_JSON OUTPUT_SVG`.
