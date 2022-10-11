# ECE1786H Assignment 2 Report

*Haocheng Wei (1008498261)*



## Section 3 Preparing the data

### 3.1 Human Data Review

1. **Review the data to see how it is organized in the file. How many examples are in the file data.tsv?**

    There are totally 10000 examples in the file.

2. **Select two random examples each from the positive set (subjective) and two from the negative set. For all four examples, explain, in English, why it has the given label.** 

    > smart and alert , thirteen conversations about one thing is a small gem . 	1
    >
    > first good , then bothersome . excellent acting and direction . 	1
    >

    These examples are marked as label 1 since they are subjective feeling of people about something (the movie they are reviewing)

    > the girls plan to take on the dons instead of buckling under their threats . 	0
    >
    > tonight there's a dancing competition at the hottest salsa restaurant in copenhagen . 	0
    >

    These examples are marked as label 0 since they are objective statements of the things and events.

3. **Find one example from each of the positive and negative sets that you think has the incorrect label, and explain why each is wrong.**

     > a moving story of determination and the human spirit . 	1
     >
     
     It should be marked as 0 since it is just a statement of the story's main rhythm.
     

     > it's charming and independent and everything hollywood is not . 	0
     >
     
     The sentence is to describe an object with the feeling it leaves to the audience. So I think it is somehow subjective. But not 100%. Actually finding subjective sentence in the negative samples is difficult since normally the description of the movie won't tell you they think their movie is "good" or "charming".


## Section 4 Baseline Model and Training

### 4.4 Overfitting to debug

**Q. Provide the training loss and accuracy plot for the overfit data in your Report.**

<img src="Report.assets/loss.png" style="zoom:67%;" /><img src="Report.assets/acc.png" style="zoom:67%;" />

### 4.5 Full Training Data

**Q. Provide the training loss and accuracy plot in your Report.**

<img src="Report.assets/nloss.png" style="zoom:67%;" /><img src="Report.assets/nacc.png" style="zoom:67%;" />

**Q. Report the final test accuracy.**

test_acc = 0.892

**Q. Answer this questions: In the baseline model, what information contained in the original sentence is being ignored?**

The positional relation in between each words in the sentences is being ignored. In the baseline model, you will get same results by using a sentence as input and using a randomly sorted same sentence as input. And that doesn't make sense to some degree.

### 4.6 Extracting Meaning from the Trained Parameters

**Q. Do some of the words that you generated make sense?**

![image-20221008223231597](Report.assets/image-20221008223231597.png)

Most of the words are adjective. They can be used to describe a movie. Most of the words make sense.

### 4.8 Submit Baseline Code

Please see `A2_Baseline.zip`, the instructions are in `Readme.md` in the zip.



## Section 5 CNN Classifier

### 5.1 Overfit

<img src="Report.assets/loss-1665470381421.png" alt="loss" style="zoom:67%;" /><img src="Report.assets/acc-1665470381420.png" alt="acc" style="zoom:67%;" />

Final Test Accuracy using Overfitting Dataset: 0.837

### 5.2 Training and Parameter Exploration

#### 5.2.1 Tuning and Model

#### 5.2.2 Unfreeze Embedding

### 5.3 Extracting Meaning from Kernels

## Section 6 Gradio

