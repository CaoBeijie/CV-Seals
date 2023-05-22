def extract_channels(df, col_ranges):
    dfs = {}
    for i, (start, end) in enumerate(col_ranges):
        dfs[f"channel_{i+1}"] = df.iloc[:, start:end]
    return dfs


#This is a random forests model.
def RandomForests_PCA_model(multi_train_x, multi_train_y, col_ranges):
    # Extract channels from multi_train_x using col_ranges
    muti_train_channel = extract_channels(multi_train_x, col_ranges)
    muti_train_HOG = muti_train_channel['channel_1']
    
    # Split the data into training and testing sets
    x_m_train, x_m_test, y_m_train, y_m_test = train_test_split(
        muti_train_HOG, multi_train_y, test_size=0.2, random_state=110, shuffle=True, stratify=multi_train_y)
    
    # Apply PCA to the training data
    pca = PCA(n_components=20)
    pca.fit(x_m_train)
    multi_pca_train_HOG = pca.transform(x_m_train)
    multi_pca_test_HOG = pca.transform(x_m_test)
    
    # Train a random forest classifier on the PCA-transformed data
    rfc_multi = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=None, min_samples_split=2)
    rfc_multi.fit(multi_pca_train_HOG, y_m_train.values.ravel())
    
    # Predict labels for the test data
    y_mlity_pre = rfc_multi.predict(multi_pca_test_HOG)
    
    # Calculate and print the accuracy of the predictions
    accuracy = accuracy_score(y_m_test, y_mlity_pre)
    #print("Multi classification accuracy of random forests: {:.2f}%".format(accuracy*100))
    
    # Return the predicted labels
    return accuracy,y_mlity_pre,y_m_test,pca,rfc_multi

#This is a regresion model.
def Regression_PCA_model(multi_train_x, multi_train_y, col_ranges):
    # Extract channels from multi_train_x using col_ranges
    muti_train_channel = extract_channels(multi_train_x, col_ranges)
    muti_train_HOG = muti_train_channel['channel_1']
    train_RGB=pd.concat([muti_train_channel['channel_3'],muti_train_channel['channel_4'],muti_train_channel['channel_5']],axis=1,join='outer')
    
    multi_train_Hog_RGB=pd.concat([train_RGB,muti_train_HOG],axis=1,join='outer')

    
    # Split the data into training and testing sets
    x_m_train, x_m_test, y_m_train, y_m_test = train_test_split(
        muti_train_HOG, multi_train_y, test_size=0.2, random_state=110, shuffle=True, stratify=multi_train_y)
    
    # Apply PCA to the training data
    pca = PCA(n_components=200)
    pca.fit(x_m_train)
    multi_pca_train_HOG = pca.transform(x_m_train)
    multi_pca_test_HOG = pca.transform(x_m_test)
    
    # Train Regression classifier on the PCA-transformed data
    lr = LogisticRegression(penalty='none',tol=0.001,class_weight='balanced',multi_class="multinomial", solver="newton-cg", max_iter=1000)
    #rfc_multi = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=None, min_samples_split=2)
    lr.fit(multi_pca_train_HOG, y_m_train.values.ravel())
    
    # Predict labels for the test data
    y_mlity_pre = lr.predict(multi_pca_test_HOG)
    
    # Calculate and print the accuracy of the predictions
    accuracy = accuracy_score(y_m_test, y_mlity_pre)
    #print("Multi classification accuracy of random forests: {:.2f}%".format(accuracy*100))
    
    # Return the predicted labels
    return accuracy,y_mlity_pre,y_m_test,pca,lr

#print resuts of Random Forests traning model.
col_ranges = [(0,900),(900, 916), (916, 932), (932, 948), (948, 964)]
bin_accuracy_rf=RandomForests_PCA_model(bin_train_x, bin_train_y, col_ranges)
multi_accuracy_rf=RandomForests_PCA_model(multi_train_x, multi_train_y, col_ranges)
multi_pre_rf=multi_accuracy_rf[1]
bin_pre_rf=bin_accuracy_rf[1]
y_m_test_rf=multi_accuracy_rf[2]
y_b_test_rf=bin_accuracy_rf[2]
balance_multi_rf=balanced_accuracy_score(y_m_test_rf, multi_pre_rf)
balance_bin_rf=balanced_accuracy_score(y_b_test_rf, bin_pre_rf)
print('----------------Training of Random forests with HOG  features-----------')
print("binary classification accuracy of Random forests : {:.2f}%".format(bin_accuracy_rf[0]*100))
print("binary classification balance of accuracy of Random forests: {:.2f}%".format(balance_bin_rf*100))

print("Multi classification accuracy of Random forests : {:.2f}%".format(multi_accuracy_rf[0]*100))
print("Multi classification balance of accuracy  of Random forests: {:.2f}%".format(balance_multi_rf*100))


#print resuts of regression traning model.
col_ranges = [(0,900),(900, 916), (916, 932), (932, 948), (948, 964)]
bin_accuracy_re=Regression_PCA_model(bin_train_x, bin_train_y, col_ranges)
multi_accuracy_re=Regression_PCA_model(multi_train_x, multi_train_y, col_ranges)
multi_pre_re=multi_accuracy_re[1]
bin_pre_re=bin_accuracy_re[1]
y_m_test_re=multi_accuracy_re[2]
y_b_test_re=bin_accuracy_re[2]
balance_multi_re=balanced_accuracy_score(y_m_test_re, multi_pre_re)
balance_bin_re=balanced_accuracy_score(y_b_test_re, bin_pre_re)
print('----------------Training of Regression with HOG features-----------')
print("binary classification accuracy of regression : {:.2f}%".format(bin_accuracy_re[0]*100))
print("binary classification balance of accuracy of regression: {:.2f}%".format(balance_bin_re*100))

print("Multi classification accuracy of regression : {:.2f}%".format(multi_accuracy_re[0]*100))
print("Multi classification balance of accuracy  of regression: {:.2f}%".format(balance_multi_re*100))


#extract traning model's PCA model and 2 fit model
bin_pca_rf=bin_accuracy_rf[3]
bin_model_rf=bin_accuracy_rf[4]
multi_pca_rf=multi_accuracy_rf[3]
multi_model_rf=multi_accuracy_rf[4]
bin_pca_re=bin_accuracy_re[3]
bin_model_re=bin_accuracy_re[4]
multi_pca_re=multi_accuracy_re[3]
multi_model_re=multi_accuracy_re[4]

#4 functions to generate test data's output CSV.
def test_Regression_bin(test_data,col_ranges):
    # Extract channels from multi_train_x using col_ranges
    muti_train_channel = extract_channels(test_data, col_ranges)
    muti_train_HOG = muti_train_channel['channel_1']
    
    multi_pca_train_HOG = bin_pca_re.transform(muti_train_HOG)
    
    # Predict labels for the test data
    y_pre = bin_model_re.predict(multi_pca_train_HOG)
    
    # Return the predicted labels
    return y_pre

def test_Regression_multi(test_data,col_ranges):
    muti_train_channel = extract_channels(test_data, col_ranges)
    muti_train_HOG = muti_train_channel['channel_1']
   
    multi_pca_train_HOG = multi_pca_re.transform(muti_train_HOG)

    y_pre = multi_model_re.predict(multi_pca_train_HOG)

    return y_pre

def test_RandomForests_bin(test_data,col_ranges):
    muti_train_channel = extract_channels(test_data, col_ranges)
    muti_train_HOG = muti_train_channel['channel_1']
   
    multi_pca_train_HOG = bin_pca_rf.transform(muti_train_HOG)

    y_pre = bin_model_rf.predict(multi_pca_train_HOG)
    
    return y_pre

def test_RandomForests_multi(test_data,col_ranges):
    muti_train_channel = extract_channels(test_data, col_ranges)
    muti_train_HOG = muti_train_channel['channel_1']
   
    multi_pca_train_HOG = multi_pca_rf.transform(muti_train_HOG)
  
    y_pre = multi_model_rf.predict(multi_pca_train_HOG)
    
    return y_pre