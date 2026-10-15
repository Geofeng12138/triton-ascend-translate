# Scan/Sort Operations


| api | brief description |
| --- | --- |
| [associative_scan](./associative_scan.md) | Applies `combine_fn` to each element of the input tensor and the carried value along the specified axis, and updates the carried value |
| [cumprod](./cumprod.md) | Returns the cumulative product of all elements in the input tensor along the specified axis |
| [cumsum](./cumsum.md) | Returns the cumulative sum of all elements in the input tensor along the specified axis |
| [histogram](./histogram.md) | Computes a histogram with `num_bins` bins based on the input tensor, each bin having a width of 1 and starting at 0 |
| [sort](./sort.md) | Sorts the tensor along the specified dimension |
| [topk](./topk.md) | Returns the top k largest elements along the specified dimension |


```{toctree}
:maxdepth: 3
:hidden:
associative_scan.md
cumprod.md
cumsum.md
histogram.md
sort.md
topk.md
```

