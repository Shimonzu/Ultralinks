# Ultralinks

Ultralinks is a web browser extension that is intended to make the web search experience richer, and more streamlined. However,
the product is experiencing a high amount of churn. My goal with this project was to make actionable recommendations to reduce
this churn. 

###Models Used
* Logistic Regression
* Ridge Regression
* Lasso Regression
* SVM
* Random Forest
* AdaBoost
* Gradient Boosting
* Survival (Kaplan-Meier, Cox Proportional Hazard)

###Tools Used
* Pandas
* Numpy
* SciPy
* SciKitLearn
* GridSearch
* StatsModels
* imblearn
* R-survival

###Process and Recommendations
I used a number of classification models to predict whether customers would churn or not. The model that generated the highest
true positive rate was a gradient boosted decision tree. From that model, I extracted the feature importances and made
actionable recommendations that will reduce churn. Here is a link to my code:

These recommendations are binned into three different categories that represent the times which Ultralinks is able to communicate
with their customers:

#####Prior to Product Download (while the user is browsing the Ultralinks website):  
1.  Make sure newWindowLinks feature defaults to OFF  
2.  Make sure replaceHyperlinks feature defaults to OFF  
3.  Emphasize the hover feature and the fact that hoverTime can be customized  
4.  Further emphasize the customizability of the Ultralinks settings  
5.  Optimize further for non-Chrome browsers  

#####During Product Download:  
1. Instruct users to set additional languages and countries  
2. Provide quick links and instructions so users can download Ultralinks to browsers other than the one in which they are
downloading the product  
3. Ask users to authenticate their Facebook, Twitter, and LinkedIn credentials so content from these platforms can be provided while using Ultralinks.  
4. Emphasize the hover feature and it's customizability again.  
5. If possible, provide a very quick video showing how to customize Ultralinks.  

#####During Update:  
1. Re-emphasize the features that each user hasn't been using, and the additional value that is provided by these features.  
2. If possible, provide again, a quick video showing how to customize Ultralinks.  


An in-depth explanation of your process
What algorithms and techniques did you use?
How did you validate your results?
What interesting insights did you gain?
Code walk-through
Give an overview of what each section of your code does.
Make it clear to the reader of your repo how they should navigate your code.
If you have a particular bit of code you think is clever or where the meat of your work is, make sure to point it out. If you tell them what to look at, they will listen.
How to run on my own
Give instructions for how to run your code on their computer (e.g. Run python scraper.py to collect the data, then run...)
