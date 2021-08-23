# telco_customer_churn
This project is about to build a binary classifier which can predict the churning customers of the Telco company. Based on a good classifier, explore the feature importance to see which factors are the biggest contributors to a churn customer. On top of that, business insights and suggestions are established and explained.

**Project Framework**

![image](https://user-images.githubusercontent.com/38795845/130504195-953c5312-b8e4-46ca-90ff-862b276a041d.png)


## Dataset Description
The dataset is obtained from Kaggle and stems from the IBM sample data set, contains 7,043 users 
and 21 features with information about whether they left the company within the last month (churn). Each 
row represents a unique costumer. These variables include: 
1. Services that each customer has signed up 
for – phone, multiple lines, internet, online security, online backup, device protection, tech support, 
streaming TV and movies
2. Customer account information – how long they have been a customer, 
contract, payment method, paperless billing, monthly charges, and total charges
3. Customer Demographic info about – gender, age range, and if they have partners and dependents, which served as 
features of the dataset, were then used to predict the churn of each user. 

![image](https://user-images.githubusercontent.com/38795845/130305907-552a22a2-04cf-4994-9d27-1b3ec2315ce9.png)

### Data Check
- Null Check: there are 11 missing values, all of them for the “total charges” column. These values are actually a blank space in the file and are exclusive for customers 
with zero tenure. It's possible to conclude that they are missing due to the fact that the customer never 
paid anything to the company. Thus, we imputed them by multiplying the tenure and monthly charges, 
since the first represents the number of months that the user was in the company, and the second indicates 
the amount paid per month. 

- Outlier Check: through the IQR, no outliers in numerical features detected with the IQR method — so no adjustments made.

- Class Balance Check: imbalanced dataset, which only consists of a 26.54% churn 
rate, dataset balancing actions should be took such like oversampling instances of the minority class, or adjust class weight proportion in hyperparameter configuration.

![image](https://user-images.githubusercontent.com/38795845/130306077-df1455cc-1009-4ecd-9216-9644301af36e.png)



##  Exploratory Data Analysis

Overall, Telco companies should know that the median tenure is about 31 months, and the median 
monthly pay is $71. As figure 1 shows, there are some features influence the churn. For example, 
churning customers have higher monthly charges with a median of $80 and a much lower interquartile 
range compared to that of non-churners (median of $65). Users are more likely to stay when the monthly 
charges are around $18 to $40. Thus, from this observation, we can recommend Telco companies review 
their pricing strategy to retain users. We will discuss the recommendations in the later sections. We also 
found that senior citizens, month-to-month contracts, without partners and children, payment method of 
electronic check, internet service of fiber optic as part of their contract are most likely churn. On the other 
hand, users tend to like online backup, device protection, and technical support as add-ons in the contract. 
Customers with these add-ons are more likely to stay with the companies. The heatmap shows electronic 
check, internet service fiber optic, tenure, and contract of two years are highly correlated to churn.

![image](https://user-images.githubusercontent.com/38795845/130305990-231e8864-b006-478e-855d-c5788dc8e9a7.png)

## Data Cleanse for Model Building

1. Removed the column that is not useful in training model such as customer ID. 
2. Values of numerical features are rescaled between a range of 0 and 1. Specifically, it should be 
standardized to have a mean of 0 and a standard deviation of 1. It is rescaled using the z-score formula. 
Because they have different scales against binary variables. 
3. Implemented one-hot encoding on 
categorical variables. It is a process by which categorical variables are converted into a form that could be 
provided to ML algorithms to do a better job in prediction. 

## Model Evaluating Metrics
In this paper, we will evaluate 3 metrics: 

- AUC score(area under ROC curve): primary consideration as it recommends models that optimize both the true positive 
and false positive rates that are significantly above random chance
- Recall on Churn customers: The ability to capture more customers going to churn, which is more worthy when we try to profile reasons for a churn. 
- Precision on Churn customers: Not too low to be accepted. Low precision means there are many retain(non-churn) cases are misclassified to churn ones, which may cause extra retention service to retain customers. That's not good for operation coast control.

Recall = True Positive / (True Positive + False Negative)

Precision = True Positives / (True Positives + False Positives)

## Methodology

1. Logistic Regression

- GridSearch is applied to search for the optimizing hyperparameters:

![image](https://user-images.githubusercontent.com/38795845/130306359-5cbfa2d8-4bb1-4a7d-bf0c-28f3967d1118.png)

- **Confusion Matrix of the Best LR Model:**

![image](https://user-images.githubusercontent.com/38795845/130306381-baeed748-7e5d-4a08-a4c0-65ce5ee0e9a6.png)

- LR Model Metrics: 

Accuracy Score:0.69 | Recall Score:0.84 | Precision Score:0.43


- ROC curve & AUC score:

AUC score: 0.8337 

![image](https://user-images.githubusercontent.com/38795845/130307534-5714b3be-06d4-40cc-91a4-d7f4a2b1aab0.png)


- Feature Importance:

![image](https://user-images.githubusercontent.com/38795845/130306520-978567c8-b4c7-48f9-8d53-87aad18df4e0.png)

- Feature Significance t-test:

![image](https://user-images.githubusercontent.com/38795845/130307610-463f73f6-1a65-4e01-8dd8-4091edc55a89.png)

For the t-test, features with P>|z| <= 0.05 are significant enough to contribute in predicting. Therefore, from the figure above,

[tenure , TotalCharges, MultipleLines_Yes, InternetService_DSL, InternetService_Fiber optic, PaperlessBilling_Yes, SeniorCitizen(0.057)] are the features should be included in profiling churn customers.

2. Decision Tree

- Best hyperparameters from GridSearch:

![image](https://user-images.githubusercontent.com/38795845/130306860-851d4f61-e241-48db-a4e5-8d006cefd843.png)


- **Confusion Matrix of Decision Tree Model:**

![image](https://user-images.githubusercontent.com/38795845/130307624-26eaaf89-36e0-44b7-93fb-b4642d090bf6.png)


- DT Model Metrics: 

Accuracy Score:0.8 | Recall Score:0.49 | Precision Score:0.60

- DT Visualization:

![image](https://user-images.githubusercontent.com/38795845/130306910-d59fa3b0-2df2-47d3-aca6-1153e1356ee1.png)

- ROC curve & AUC score:

AUC score: 0.8337 

![image](https://user-images.githubusercontent.com/38795845/130306985-62e1a6e4-8245-4f62-b166-3a4d20a0796c.png)

3. Kth Nearest Neighbor Model

- Find the best K:

![image](https://user-images.githubusercontent.com/38795845/130307745-ed6c6bee-79a1-454b-be6d-7c191532fb5e.png)

K=38 looks a best k for this classifier.

- **Confusion Matrix of KNN Model:**

![image](https://user-images.githubusercontent.com/38795845/130307801-35a95b10-9cdd-447f-b974-113aa4eb7114.png)


- KNN Model Metrics: 

Accuracy Score:0.78 | Recall Score:0.52 | Precision Score:0.55

- ROC curve & AUC score:

AUC:0.8047

![image](https://user-images.githubusercontent.com/38795845/130307833-02c3d862-d502-4015-9f3f-e7074056a6a8.png)


4. Random Forest 

- Best hyperparameters from GridSearch:

![image](https://user-images.githubusercontent.com/38795845/130307284-90f97136-c733-469c-8836-e789884c4207.png)

RandomForestClassifier(class_weight='balanced', criterion='entropy',
                       max_depth=6, min_samples_leaf=5, min_samples_split=0.1,
                       n_estimators=50, n_jobs=-1, random_state=19)

- **Confusion Matrix of Random Forest Model:**

![image](https://user-images.githubusercontent.com/38795845/130307257-4a6d965c-35de-4a48-bb64-8f340c2d5812.png)


- RF Model Metrics: 

Accuracy Score:0.74 | Recall Score:0.77 | Precision Score:0.78

- ROC curve & AUC score:

AUC: 0.85

![image](https://user-images.githubusercontent.com/38795845/130307342-5dd17ea4-dd25-41d7-b4c3-e12c69d29dd0.png)

- Feature Importance of RF model:

![image](https://user-images.githubusercontent.com/38795845/130307881-21239037-cefc-46a1-a5d9-8f71a4088e00.png)

we found that tenure, contract duration terms,
additional services and payment method seem to be among the most important drivers of churn. This 
makes a lot of sense intuitively: the longer the contract duration the less likely it is that the customer will 
churn as he/she is less frequently confronted with the termination/prolongation decision and potentially 
values contracts with reduced effort. We also see that customers with internet service fiber optic as part of 
their contract influences the churn again, which is the same observation as Logistic Regression. Thus, the 
company may need to review their internet service fiber optic service to see if there has any problems, 
which we will discuss the recommendations later. 


5. Nonlinear Support Vector Machine

The underlying principle behinds nonlinear SVM classifier is using kernel tricks with linear 
SVM. The kernel technique is divided into two steps: map the data from the original space to a higher dimensional or even infinite-dimensional space; then use a linear classifier to fit the data in the new space. 
This model is applied for checking if mapping data to higher dimension space improve prediction 
accuracy. The Gaussian kernel (i.e radial basis function kernel (‘rbf’)) is used in this task. Noted that RBF 
kernel does not provide a prediction for the importance or coefficients of variables.

- **Confusion Matrix of Nonlinear SVM Model:**

![image](https://user-images.githubusercontent.com/38795845/130307656-5f65410a-ff9f-475d-87fe-9b8792a838a0.png)


- SVM Model Metrics: 

Accuracy Score:0.80 | Recall Score:0.52 | Precision Score:0.60

- ROC curve & AUC score:

AUC score: 0.8271

![image](https://user-images.githubusercontent.com/38795845/130307678-7def309d-48ef-48e8-9634-e612a3a03122.png)



## Performance Comparison

Metrics Evaluation:

![image](https://user-images.githubusercontent.com/38795845/130308041-0771d9bd-0ca4-453f-92e4-fc4fdb5447cb.png)

The importance of this type of research in the telecom market is to help companies make more 
profit. It has become known that predicting churn is one of the most important sources of income to 
telecom companies. Hence, this research aimed to build a model that predicts the churn of customers and 
need to achieve high AUC values. We have implemented Logistic Regression, Decision Tree, KNN, SVM, and Random 
Forest, and discovered that Random Forest performs best among all models with the highest AUC 85%. Although Random Forest has a drawback that it can be slow to process data as they are computing data for each individual decision tree, considered its higher AUC and second highest recall, we still suggest 
Random Forest would be the best model to predict customer churn. 

## Suggestion & Conclusion

Looking at the evaluation results, 
specifically the feature weights from these 5 models, internet services fiber optic, contract duration 
month-to-month, and monthly/total charges are the biggest driver of churn. 
- If telco company wants to retain their users, should control their monthly charges to $18 to $40, or providing discount
regularly/seasonally to retain users. The fiber optic has increased customer churn and this could be due 
to price for the option or the internet speed is not stable. As a next analysis, we might want to compare 
the price and speed performance with that of competitors who provide similar service

#### Further Suggestion on Churn Risk Classification

Balancing precision and recall for actionable retention tactics should be taken. By 
moving the decision threshold probability along the precision-recall curve, we may find tranches of 
churn cases where we feel confident enough to deploy real retention actions. By plotting precision and 
recall scores as a function of decision threshold (DT), we can see the probability required to achieve the 
precision we want. We recommend splitting the model into a 3 tier approach, covering these incremental 
churn cases. As an example from figure 10 of Precision-Recall scores vs. DT of Random Forest, we 
could set users to 3 tiers of churn risk. 

![image](https://user-images.githubusercontent.com/38795845/130308162-fa70616c-ed38-42d7-80e9-f3701bf7c21c.png)

- When the DT is set at 0.75 / 80% Precision / 20% Recall, 8 in 10 predicted churn customers will churn, 2 in 10 churn customers are captured.
user’s churn level is high risk, indicating nearly every positive sample is true (churn). Company should 
target most aggressive retention like proactive outreach with discounts or adding free services to retain 
the mostly likely-to-churn customers. 

- When DT is set at 0.65 / 70% Precision / 40% Recall, 7 in 10 predicted churn cases will churn, 4 in 10 churn customers are captured.
users’ churn level is medium risk. Company can target more general, and lower cost, approaches like 
empowerment of customer service agents to save customers in certain contexts.

- When DT is set at 0.5 / 50% Precision / 80% Recall, 5 in 10 predicted churn customers will churn, 8 in 10 churn customers are captured.
 the negative samples are least likely to churn. Company can add 
these users to watch lists, or general retention communications efforts. 

Finally, further improvement can 
be achieved by introducing more variables such as user life time value, which can be done by 
multiplying tenure and monthly charges. So that we can predict the false negative cost, which would 
give telco companies more insights about the cost optimization and risk analysis. Overall, this project 
provides a strong predictive model that will help telco companies to gain more understanding of churn 
factors and help them improve their services to retain users. 









