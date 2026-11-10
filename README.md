# Edge-of-stability convergence rates on the diabetes dataset

Depth-5, width-1 linear network fitted to standardized BMI (n = 442) with gradient descent at ηλ* = 1.975, 2.000, 2.025, and with minibatch SGD at batch sizes 32, 64, 128, 221 (20 seeds each).

```
pip install -r requirements.txt
python run.py
```

Runs in about 2.5 minutes on one CPU and writes `results.json`, `fig_fullbatch.png` and `fig_batch32.png`.
