# Adding Datasets

The datasets used here are samples extracted from the [Amazon Customer Reviews][datasets] data.

It is assumed that `star_rating`s of 1 & 2 imply negative reviews, whereas those greater than 3 are positive.

To fetch more data, download the files from <https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt>.

You can then run the handy Python script:

```bash
python add-sample-datasets.py
```

to process them.

[datasets]: https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt
