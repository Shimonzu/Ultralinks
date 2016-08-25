# Ultralinks

Ultralinks is a web browser extension that is intended to make the web search experience richer, and more streamlined. However,
the product is experiencing a high amount of churn. My goal with this project was to make actionable recommendations to reduce
this churn. 

Unfortunately, due to the proprietary nature of the Ultralink data, I'm not able to share that, but please see the process description below as well as my code used to process the data.

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
actionable recommendations that will reduce churn. Click on the [Ultralink_Code](https://github.com/Shimonzu/Ultralinks/blob/master/Ultralinks_Code.py) file to see my code:

Because there is no requirement to create an Ultralinks account, or provide an email address in order to download the Ultralinks product, there is no way for Ultralinks to communicate with it's customers except on the Ultarlinks website, during product download, and during subsequent product updates. For that reason, these recommendations are binned into three different categories that represent the times which Ultralinks is able to communicate
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
