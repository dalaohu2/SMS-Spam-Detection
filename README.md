## overview
Nowadays, scam text messages disguise by characters rather than letters, so I set out to test whether  a standard text catergory model can recognise the scam text messages with disguise.I have trained a a catergory model of TF-IDF+Logistic Regression  trained on 5,572 text messages. It reached  96.8% accuracy, yet the figure is misleading--the model caught only 76% of real spam. A simple leetspeak disguise make the recall drop to 19%, but these messages still are fraudulent to a human reader. This project is the second one in an adversarial ML series, the first one is phishing detection, the same attack routine transfers across both numeric features and text.
## Dataset
There is heavily imbalanced between the normal text messages and spam,4,825 and 747, respectively,only 13% are spam. The total data contains 5,572 messages.The database is the SMS Spam Collection（UCI）.
## Model
TF-IDF + Logistic Regression
## Evaluation: why accuracy lies
The model reached 96.8% accuracy, yet this number is misleading because data's class is heavily imbalanced.A model that labelled every message as legitimate would still reach 86.6% accuracy, so accuracy alone reveals little. The model is 100% right when text messages flag as spam, but it suffers from a high false-negative rate(24%).Every error it makes is a false negative; it raises no false alarms. In spam filtering this is the dangerous direction, since a missed scam can defraud the user.
## Adversarial test
I wrote a disguise function by using leetspeak to change characters, for example, "free" turning to be "fr33".The original model caught spam（113 out of 149），but it only caught 29 out of 149 after disguises. The disguises could be recognised as spam by human reader at a glance. And the reason of it is that TF-IDF just know the same word form the vocabulary list, which means "fr33" becomes the out-of-vocabulary. So that its weight drops to zero, and the model loses the signal it relied on.
## Limitations & Defense
Limitation: the model just relies on the vocabulary list, which means a tiny  disguise could evade it.Even on clean data, it miss 24% of real spam. To fix it , I have two methods.
Method A: adversarial training--feed disguised samples back into retrain.
Method B: character n-grams--let model watch the character pieces.so that "fr33" still shares pieces like "fr" with "free"
But hardening it is like robustness-accuracy trade-off.
## Connection
It is the second project in adversarial ML series(the first one is phishing detection). In both programs, models rely on only a few signals. If I change the signals, the dtection would collapses. So, the vulnerablity is not accident,but failure mode of transferring across domains.
