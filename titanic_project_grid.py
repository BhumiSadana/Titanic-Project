import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

df=pd.read_csv("titanic.csv")
print(df.head(5))

print(df.isnull().sum())

print(df.dtypes)

df["Age"].fillna(0,inplace=True)
df["Cabin"].fillna("None",inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)



num_feature=["Age","Fare"]
cat_feature=["Sex","Embarked"]

X=df[num_feature+cat_feature]
y=df["Survived"]

preprocessor=ColumnTransformer([
    ('num',StandardScaler(),num_feature),  #feature scaling
     ('cat',OneHotEncoder(),cat_feature)
    ])

pipeline=Pipeline([
    ('preprocessor',preprocessor),
    ('classifier',RandomForestClassifier(random_state=42))
    ])


'''params_dist={
    'classifier__n_estimators':[50,100,200],
    'classifier__max_depth':[50,10,None],
    'classifier__min_samples_split':[2,5]
    }'''

params_dist={
    'classifier__n_estimators':randint(50,501),
    'classifier__max_depth':randint(2,21),
    'classifier__min_samples_split':randint(2,21)
    
    }

#grid=GridSearchCV(pipeline,params_dist,cv=5,scoring="accuracy")
random_search=RandomizedSearchCV(pipeline,params_dist,n_iter=10,cv=5,scoring="accuracy",random_state=42)



#train_test

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
#model fit
'''grid.fit(X_train,y_train)
print("best params:",grid.best_params_)
print("best score:",grid.best_score_)'''

random_search.fit(X_train,y_train)
print("best score:",random_search.best_score_)

#predict
#y_pred=grid.predict(X_test)
y_pred=random_search.predict(X_test)

#evaluate

print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion_matrix:",confusion_matrix(y_test,y_pred))
print("Classification_report:",classification_report(y_test,y_pred))


