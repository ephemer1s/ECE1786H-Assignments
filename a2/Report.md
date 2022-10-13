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

<img src="Report.assets/loss.png" style="zoom: 40%;" /><img src="Report.assets/acc.png" style="zoom:40%;" />

### 4.5 Full Training Data

**Q. Provide the training loss and accuracy plot in your Report.**

<img src="Report.assets/nloss.png" style="zoom: 40%;" /><img src="Report.assets/nacc.png" style="zoom:40%;" />

**Q. Report the final test accuracy.**

test_acc = 0.892

**Q. Answer this questions: In the baseline model, what information contained in the original sentence is being ignored?**

The positional relation in between each words in the sentences is being ignored. In the baseline model, you will get same results by using a sentence as input and using a randomly sorted same sentence as input. And that doesn't make sense to some degree.

### 4.6 Extracting Meaning from the Trained Parameters

**Q. Do some of the words that you generated make sense?**

<img src="Report.assets/image-20221008223231597.png" alt="image-20221008223231597" style="zoom:50%;" />

Most of the words are adjective. They can be used to describe a movie. Most of the words make sense.

### 4.8 Submit Baseline Code

Please see `A2_Baseline.zip`, the instructions are in `Readme.md` in the zip.



## Section 5 CNN Classifier

### 5.1 Overfit

<img src="Report.assets/loss-1665470381421.png" alt="loss" style="zoom:40%;" /><img src="Report.assets/acc-1665470381420.png" alt="acc" style="zoom:40%;" />

Final Test Accuracy using Overfitting Dataset: 0.837

### 5.2 Training and Parameter Exploration

#### 5.2.1 Tuning and Model

In this section and the following section, Grid Search is used in tuning the model hyperparameters. By using `python A2_CNN/main.py -g 1 -f 1`, Grid Search method is used and normal arguments are blocked. Argument from `A2_CNN/gs_args.json` is imported instead. 

When using Grid Search, the parameters tuned are $k_1$, $k_2$, $n_1$, $n_2$ and $lr$ (learning rate). We assume that $k_1 \leq k_2$ to cut off abundant cases.  And we use 30 epochs, validate the model per 3 epochs to save tuning time. Normally, the tuning time on RTX 2060 GPU will be 30~60 seconds per cases.

The model achieved **best test accuracy of 91.45%** with the following params: $[k_1, k_2, n_1, n_2, lr] = [2, 4, 4, 16, 0.0002]$.

The curves are as follows:

<img src="Report.assets/loss-1665522015023.png" alt="loss" style="zoom:40%;" /><img src="Report.assets/acc-1665522021533.png" alt="loss" style="zoom:40%;" />

#### 5.2.2 Unfreeze Embedding

Using `python A2_CNN/main.py -s 1` to train unfrozen model.

The accuracy is 91.15%. Which is slightly lesser than model with unfrozen embedding layer.

### 5.3 Extracting Meaning from Kernels

Firstly, I think it remains a question whether each kernels from different convolution networks are weighted equal or weighted by the portion of the kernel in its network.

As we calculated the 5 closest words of each of the 72 kernels, we added up all the similarity value of the top words.

The result shows the top 5 closest words to all 72 kernels and their scores are

```python
('abducted', 2.3963323), 
('flees', 2.4324336), 
('fiancée', 2.475415), 
('chintzy', 2.828094), 
('fiancé', 3.478285)
```

Which seems like it doesn't make sense.

While we look into the kernels separately, we found that each kernel has different set of closest words indicating different categories.

<img src="Report.assets/image-20221011172623111.png" style="zoom:50%;" /><img src="Report.assets/image-20221011172638675.png" style="zoom:50%;" /><img src="Report.assets/image-20221011172650511.png" style="zoom:50%;" /><img src="Report.assets/image-20221011172712397.png" style="zoom:50%;" />

From these set of words, we can infer that each kernels learned different pattern of words that can determine the subjectivity of all sentences, or at least, some sentences.

## Section 6 Gradio

### 6.1 Run and Compare

Notes: users may encounter errors when the input sentence is shorter than kernel size.

#### Models used:

* **model 1:** Baseline model. Accuracy: 89.2%
* **model 2:** CNN Classifier, $[k_1, k_2, n_1, n_2] = [2, 5, 16, 32]$, unfrozen. Accuracy: 91.8%

#### Examples and Results:

* **Strong Subjective:** I think this course has too much workload in assignments. 
  * All correct. Model that have accuracy of 90% shouldn't make mistake on this strong cases.
* **Weak Subjective:** Jollibee has the best fried chicken in Toronto that I have ever seen.
  * All correct. Actually I'm not sure if this sentence is subjective. But as the models both agree that this is subjective, I will listen to them.
* **Weak Objective:** "I think this sentence is objective" is not an objective sentence.
  * Model 2 correct. This is a tricky one. "I think" makes the kernel goes to subjective side. Model 2 makes the correct prediction while model 1 is baited.
* **Strong Objective:** The Roman Empire falls in the year 1991 A.D.
  * All correct. Model that have accuracy of 90% shouldn't make mistake on this strong cases. Although the sentence it self is incorrect.

To conclude, the model 2 performs best among all the current and previous models. It runs fast, gives the correct prediction on the very tricky "Weak Objective" example, and has the best accuracy on the test dataset.