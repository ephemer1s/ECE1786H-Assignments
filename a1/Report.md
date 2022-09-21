# Assignment 1 Answers

## Section 1

### S1.2

Provide a table that compares the 10-most cosine-similar words to the word ‘dog’, in order, alongside to the 10 closest words computed using euclidean distance. 

![](Report.assets/image-20220919003048386.png)![](Report.assets/image-20220919003102973.png)

(Left: CosSim Right: ED)

Give the same kind of table for the word ‘computer.’ 

![](Report.assets/image-20220919003120175.png)![](Report.assets/image-20220919003132648.png)

Does one of the metrics (cosine similarity or euclidean distance) seem to be better than the other? 

No. The result above does not provide enough evidence of one metric better than the other.

1. From the result we can see that there is very slight difference between the 10-most similar words calculated by different metrics.
2. Each words in the list certainly seems to match the requirement to be similar to the original word.
3. The gap between numerical scores are slightly bigger using ED than using CosSim. But I don't think this make ED a more accurate metric. Because using ED you get score in $[0, +\infin)$ while using CosSim you get $[-1, 1]$



### S1.3 

In `A1P1_4.py`, I choose to convert cities to their countries. The result is as the following table shows.

| Cities  |                            Result                            |   Cities   |                            Result                            |
| :-----: | :----------------------------------------------------------: | :--------: | :----------------------------------------------------------: |
| Beijing | ![image-20220920170725668](Report.assets/image-20220920170725668.png) |   Athens   | ![image-20220920170830694](Report.assets/image-20220920170830694.png) |
|  Tokyo  | ![image-20220920170738572](Report.assets/image-20220920170738572.png) |   Ottawa   | ![image-20220920170840100](Report.assets/image-20220920170840100.png) |
|  Seoul  | ![image-20220920170803604](Report.assets/image-20220920170803604.png) | Leningrad  | ![image-20220920170848252](Report.assets/image-20220920170848252.png) |
| London  | ![image-20220920170810362](Report.assets/image-20220920170810362.png) | Washington | ![image-20220920170856811](Report.assets/image-20220920170856811.png) |
|  Paris  | ![image-20220920170816069](Report.assets/image-20220920170816069.png) |   Riyadh   | ![image-20220920170904062](Report.assets/image-20220920170904062.png) |

The pattern $athens - city + nation = greece$ is very precise to most of the city in the list while using CosSim. We can confirm that the quality of the result is better than expected: it gives "USSR" for "Leningrad". However, there are some anomalies. For "Washington", "U.S." is on the 5th place in the list. I think the reason is the other meanings of it weight more than "Washington D.C."

### S1.4

I think this may support that vectors have bias in ethics of origin, but the evidence is not very strong. As I haven't found any article mentioning this.

![image-20220920214144538](Report.assets/image-20220920214144538.png)

### S1.5

How does the euclidean difference change between the various words in the notebook when switching from d=50 to d=300? 

How does the cosine similarity change? 

Does the ordering of nearness change? 

Is it clear that the larger size vectors give better results - why or why not?

### S1.6

State any changes that you see in the Bias section of the notebook.





## Section 2





## Section 3