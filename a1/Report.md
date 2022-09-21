# Assignment 1 Answers

## Section 1

### S1.2

* Provide a table that compares the 10-most cosine-similar words to the word ‘dog’, in order, alongside to the 10 closest words computed using euclidean distance. 

![](Report.assets/image-20220919003048386.png)![](Report.assets/image-20220919003102973.png)

(Left: CosSim Right: ED)

* Give the same kind of table for the word ‘computer.’ 

![](Report.assets/image-20220919003120175.png)![](Report.assets/image-20220919003132648.png)

* Does one of the metrics (cosine similarity or euclidean distance) seem to be better than the other? 

  No. The result above does not provide enough evidence of one metric better than the other.

  1. From the result we can see that there is very slight difference between the 10-most similar words calculated by different metrics.
  2. Each words in the list certainly seems to match the requirement to be similar to the original word.
  3. The gap between numerical scores are slightly bigger using ED than using CosSim. But I don't think this make ED a more accurate metric. Because using ED you get score in $[0, +\infin)$ while using CosSim you get $[-1, 1]$



### S1.3 

In `A1P1_4.py`, I choose to convert cities to their countries. The result is as the following table shows.

| Cities | Result | Cities | Result |
| ------ | ------ | ------ | ------ |
|        |        |        |        |
|        |        |        |        |
|        |        |        |        |
|        |        |        |        |
|        |        |        |        |