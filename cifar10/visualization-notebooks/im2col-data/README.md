Filename follows this naming scheme:

```
file_name_filters = 
"im2col-data/filters-bs{}-h{}-w{}-c{}-k{}-d{}-bid{}-sbid{}-cid{}-iters{}".format(
    batch_size,
    in_height,
    in_width,
    n_channels,
    k_size,
    n_filters, # d -> number of output channels
    block_id,  # bid -> id of block of convs with same channels
    sublock_id, # number of pairs of convs
    conv_id,    # first or second conv of the pair
    iter_number
)
```


Only the last convs of each block have been collected.

Each file has a different sparsity. Better check.
Here is an example for iteration 100

From the last block to the first
```
conv4_1/conv2_in_block/Relu/sparsity_histo
Matrix A: Filters ((512, 4608)), sparsity = 0.0
Matrix B: Images ((4608, 16)), sparsity = 0.6709255642361112

conv3_1/conv2_in_block/Relu/sparsity_histo
Matrix A: Filters ((256, 2304)), sparsity = 0.0
Matrix B: Images ((2304, 64)), sparsity = 0.6403266059027778

conv2_1/conv2_in_block/Relu/sparsity_histo
Matrix A: Filters ((128, 1152)), sparsity = 0.0
Matrix B: Images ((1152, 256)), sparsity = 0.5699801974826388

conv1_1/conv2_in_block/Relu/sparsity_histo
Matrix A: Filters ((64, 576)), sparsity = 0.0
Matrix B: Images ((576, 1024)), sparsity = 0.5332828097873263
```
