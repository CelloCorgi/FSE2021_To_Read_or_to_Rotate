import pandas as pd  
import numpy as np
from scipy.stats import ttest_ind, ttest_rel, pearsonr, spearmanr  
import matplotlib.pyplot as plt  
import seaborn as sns

"""
This script makes a correlation matrix of our major assessment scores
"""

preYoon = "Pre-test: Yoon (Spatial X / 30)"
prePFT = "Pre-test: PFT (x / 20)"
preBoth = "Both Pre spatial tests (added)"
preReading = "Pre-test: Reading"
postYoon = "Post-test: Yoon"
postReading = "Post-test: Reading"
postPFT = "Post-test PFT"
postProg = "Post-test: Programming (x / 27)"
defProg = "Post-test: Programming Def"
traceProg = "Post-test: Programming Trace"
codeProg = "Post-test: Programming Code"
preProg = "Pre-test: Programming (x / 12)"
deltaProg = "Programming Delta"

# Function below from https://machinelearningmastery.com/effect-size-measures-in-python/
# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s


def corr_sig(df=None):
    """
    From: https://stackoverflow.com/questions/57226054/seaborn-correlation-matrix-with-p-values-with-python
    """
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    r_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    all_pts = []
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            r, p = spearmanr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
            r_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = r
            all_pts.append((p, r))

    # This part prints out what I need for my false comparsion rate
    # Using FDR BY correction
    all_pts=sorted(all_pts)
    current = 1
    m = 91 # Number of comparisons
    q = 0.05 # FDR rate
    q_prime = q / sum(( 1 / i for i in range(1, m))) # Calculated for BY comparsion
    print(q_prime)
    #first = True
    p_cutoff = 0.05
    for x in reversed(all_pts):
        stat = ((current / 2) / (m)) * q_prime
        print("{}: {}, {}".format(current, x, stat))
        if x[0] > stat:
            if x[0] < p_cutoff:
                p_cutoff = x[0]
                print("All above this not significant!")
        current += 1
    print(p_cutoff)
    return p_matrix, p_cutoff



def plot_correlation_matrix(df):
    df_corr = df_final
    #Pre-test: PFT (x / 20),Post-test PFT,Pre-test: Yoon (Spatial X / 30),Post-test: Yoon,Pre-test: Reading,Post-test: Reading,Pre-test: Programming (x / 12),Post-test: Programming (x / 27),Gender,Ethnicity,Native Language,IDE,Expected EECS 183 grade,Planing on Majoring / Minoring in CS/DS/SI,Study effected plans to continue in CS ,Feel Personly study helped,
    df_corr.drop(["Programming Delta",
                "Both Pre spatial tests (added)", 
                "Both_delta",
                "tt_Spatial", 
                "tt_Reading", 
                "Reading_delta", 
                "PFT_delta", 
                "Yoon_delta", 
                "natlang_Arabic", 
                "natlang_Spanish", 
                "gender_Male", 
                "natlang_Korean", 
                "natlang_Bengali", 
                "natlang_Urdu", 
                "Post-test: Programming (x / 12)", 
                "Pre-test Reading Version", 
                "Participant ID", 
                "Attended Post-test", 
                "Num Training Sessions", 
                "Participant Uniquename", 
                "Follow-up?", 
                "Store Data", 
                "Socioeconomic Status", 
                "Planing on Majoring / Minoring in CS/DS/SI", 
                "Ethnicity", 
                "IDE","Expected EECS 183 grade",
                "Feel Personly study helped", 
                "Study effected plans to continue in CS "], axis=1, inplace=True)
    df_corr = df_corr.rename(columns={prePFT: "Pre PFT (1)", 
                                    postPFT: "Post PFT (2)", 
                                    preYoon: "Pre PSVT:R II (3)", 
                                    postYoon: "Post PSVT:R II (4)", 
                                    preReading: "Pre Reading (5)", 
                                    postReading: "Post Reading (6)", 
                                    preProg: "Pre SCS1 (7)", 
                                    postProg: "Post SCS1 (8)",
                                    defProg: "Post SCS1 Def (9)",
                                    traceProg: "Post SCS1 Trace (10)",
                                    codeProg: "Post SCS1 Code (11)",
                                    "gender_Female": "Gender Female (12)",
                                    "natlang_Chinese": "Lang Chinese (13)",
                                    "natlang_English": "Lang English (14)"})
    p_values, p_cutoff = corr_sig(df_corr)
    mask = np.invert(p_values < p_cutoff)
    plt.figure(figsize=(10,12))
    cor = df_corr.corr(method="spearman", min_periods=5)
    sns.set(font_scale=2.25)
    x_lables = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14" ]
    sns.heatmap(cor, annot=True,  mask=mask, cmap='RdBu_r', xticklabels=x_lables, linewidths=0.5, linecolor="grey")
    sns.set(style="darkgrid", color_codes=True)
    plt.show()


def preprocess_data(raw_data, num_sessions):
    # This function is a place for filtering and one-hot encoding. We first encode for training type
    dataset = pd.concat([raw_data, pd.get_dummies(raw_data['Training Type'], prefix='tt')],axis=1)
    dataset.drop(['Training Type'],axis=1, inplace=True)
    # Now one-hot encode for gender
    dataset = pd.concat([dataset, pd.get_dummies(dataset['Gender'], prefix='gender')],axis=1)
    dataset.drop(['Gender'],axis=1, inplace=True)
    # Now one-hot encode for native language
    dataset = pd.concat([dataset, pd.get_dummies(dataset['Native Language'], prefix='natlang')],axis=1)
    dataset.drop(['Native Language'],axis=1, inplace=True)
    # Filter the data just for those being included in this analysis
    did_post_test = dataset['Attended Post-test']==True
    dataset_post_test = dataset[did_post_test]
    print(dataset_post_test.shape)
    # Now filter for the number of sessions
    enough_sessions = dataset_post_test['Num Training Sessions']>=num_sessions
    return dataset_post_test[enough_sessions]

def get_just_spatial_and_just_reading(data):
    spatial_f = data["tt_Spatial"] == 1
    reading_f = data["tt_Reading"] == 1
    return data[spatial_f], data[reading_f] 

def add_column_differences(preprocessed_data):
    preprocessed_data["Yoon_delta"] = preprocessed_data["Post-test: Yoon"] - preprocessed_data[preYoon]
    preprocessed_data["PFT_delta"] = preprocessed_data["Post-test PFT"] - preprocessed_data[prePFT]
    preprocessed_data["Reading_delta"] = preprocessed_data["Post-test: Reading"] - preprocessed_data["Pre-test: Reading"]
    preprocessed_data["Both_delta"] = preprocessed_data["Yoon_delta"] + preprocessed_data["PFT_delta"]
    preprocessed_data[preBoth] = preprocessed_data[preYoon] + preprocessed_data[prePFT]
    preprocessed_data[deltaProg] = preprocessed_data[postProg] - preprocessed_data[preProg]

    return preprocessed_data

def filter_gender(preprocessed_data):
    female_participants = preprocessed_data['gender_Female']>0
    male_participants = preprocessed_data['gender_Male']>0
    return preprocessed_data[female_participants], preprocessed_data[male_participants]

def filter_native_lang(preprocessed_data):
    English_native = preprocessed_data['natlang_English'] > 0
    Chinese_native = preprocessed_data['natlang_Chinese'] > 1
    non_english = preprocessed_data['natlang_English'] < 1

    return preprocessed_data[English_native], preprocessed_data[Chinese_native], preprocessed_data[non_english]

def filter_low_incoming_spatial(preprocessed_data, value):
    low_spatial = preprocessed_data[preBoth] < value
    high_spatial = preprocessed_data[preBoth] > value
    return preprocessed_data[low_spatial], preprocessed_data[high_spatial]

def filter_low_incoming_reading(preprocessed_data, value):
    low_reading = preprocessed_data[preReading] <= value
    high_reading = preprocessed_data[preReading] > value
    return preprocessed_data[low_reading], preprocessed_data[high_reading]


if __name__ == "__main__":
    dataset = pd.read_csv('StudyData/data3.csv')
    df_preprocessed = preprocess_data(dataset, 6) 
    df_final = add_column_differences(df_preprocessed)
    df_spatial, df_reading = get_just_spatial_and_just_reading(df_final)
    reading_f, reading_m = filter_gender(df_reading)
    spatial_f, spatial_m = filter_gender(df_spatial)

   
    #Here we will print some averages
    average_female_final_spatial = np.average(spatial_f[postProg])
    print("Average female spatial post-programming score: {}".format(average_female_final_spatial))

    average_female_final_reading = np.average(reading_f[postProg])
    print("Average female reading post-programming score: {}".format(average_female_final_reading))

    average_male_final_spatial = np.average(spatial_m[postProg])
    print("Average male spatial post-programming score: {}".format(average_male_final_spatial))

    average_male_final_reading = np.average(reading_m[postProg])
    print("Average male reading post-programming score: {}".format(average_male_final_reading))
    #plot_box_wisker_plot(df_reading, df_spatial)
    plot_correlation_matrix(df_final)

