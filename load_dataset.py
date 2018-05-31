import re
import csv
import time
# import pickle
import os, sys
import itertools
from utils import *
import numpy as np
import pandas as pd
from numba import jit
import seaborn as sns
from scipy import stats
import cPickle as pickle
from fuzzywuzzy import fuzz
from termcolor import cprint
from collections import Counter
import matplotlib.pyplot as plt
from pprint import pprint, pformat
from colnames import date_colnames

# fuzz is used to compare TWO strings
from fuzzywuzzy import fuzz
# process is used to compare a string to MULTIPLE other strings
from fuzzywuzzy import process

from sklearn import preprocessing
from sklearn.utils import shuffle as skl_shuffle
from sklearn.model_selection import train_test_split

from us_state_abbrev import us_state_abbrev

employer_sizes = [0, 5, 10, 50, 100, 2000, 20000, 100000]
employer_size_labels = ['pico', 'nano', 'micro', 'milli', 'medium', 'kilo', 'mega', 'giga']

def min_max_norm(s):
    return (s - np.min(s)) / (np.max(s) - np.min(s))

def norm_zscore(s):
    return (s - np.mean(s)) / np.std(s)

def discretise_employer_size(value):
    out_label = 'nan'
    for size, label in zip(employer_sizes, employer_size_labels):
        if value >= size:
            out_label = label

    # print(value)
    # print(out_label)
    return out_label

# @jit
# def find_possible(possibilities, min_ratio):
    
#     output = []
#     for item, score in possibilities:
#         if score > min_ratio:
#             output.append((item, score))

#     return output
        

def fuzzy_match_and_combine(s, min_ratio=90):
    s = s.map(lambda x: x.upper().replace('.', '').replace(',', '').replace(' AND ', ' & ') if type(x) == str else x)
    s_filtered = s.copy()

    try:
        with open(s.name + '_fuzz.pkl', 'rb') as f:
            fuzzy_mapping = pickle.load(f)
    except IOError as e:
        print(e)
        fuzzy_mapping = {}

    cprint('Loaded {}'.format(len(fuzzy_mapping.keys())), 'white', 'on_magenta')

    # items = []
    # for item in s.tolist():
    #     try:
    #         items.append(str(item).encode())
    #     except Exception as e:
    #         print(e)

    # items = [str(i) for i in ]
    # print(len(items))
    # counts = Counter(items)

    firm_names_unmapped = list(set([str(i) for i in s.tolist()]) - set(fuzzy_mapping.keys()))
    # fuzzy_mapping = {}

    # unique_items = list(set(items))

    # fuzz_mapping = {}
    # for item in unique_items:
    #     possibilities = process.extractWithoutOrder(item, unique_items, scorer=fuzz.ratio)
    #     start = time.time()
    #     possible = [possible for possible in possibilities if possible[1] > min_ratio]
    #     print(time.time() - start)
    #     match, sim = sorted(possible, key=lambda x: counts[x[0]], reverse=True)[0]
    #     print(item, match, sim)
    #     fuzz_mapping[item] = match

    ############################## UNCOMMENT FOR INFERENCE ##########################
    match_count = 0
    for name1, name2 in itertools.combinations(firm_names_unmapped, 2):
        ratio = fuzz.ratio(name1, name2)
    #     ratio = SequenceMatcher(None, "abcd", "bcde")

        if ratio >= min_ratio:
            match_count += 1
            counts = Counter(s)
            print(match_count, name1, name2, ratio, counts[name1], counts[name2])

            if counts[name1] >= counts[name2]:
                fuzzy_mapping[name2] = name1
            else:
                fuzzy_mapping[name1] = name2
        else:
            # If no match, name maps to itself
            fuzzy_mapping[name1] = name1
    #################################################################################

    with open(s.name + '_fuzz.pkl', 'wb') as f:
        print('saving pkl')
        pickle.dump(fuzzy_mapping, f)

    return s_filtered.map(lambda x: fuzzy_mapping[x] if (type(x) == str and (x in fuzzy_mapping)) else x)

def load_and_preprocess(path, fuzzy_matching=True):
    # df_test = pd.read_csv('TestingSet_Random(1).csv')
    # df_test2 = pd.read_csv('TestingSet(2).csv')
    df = pd.read_csv(path)

    print(len(df.index))
    # print(list(df))
    # print(df['case_status'])

    # FIXME
    if fuzzy_matching:
        # df['employer_name_modified'] = df.employer_name.map(lambda x: re.sub(r'([^\s\w]|_)+', '', x.lower()) if (type(x) == str) else x)
        # df['agent_firm_name_modified'] = df.agent_firm_name.map(lambda x: re.sub(r'([^\s\w]|_)+', '', x.lower()) if (type(x) == str) else x)
       df['agent_firm_name_modified'] = fuzzy_match_and_combine(df.agent_firm_name, 91)

        # print(df['agent_firm_name_modified'].value_counts())
        # exit()
    else:
        df['agent_firm_name_modified'] = df['agent_firm_name']

#     df['agent_state_abbr'] = df.agent_firm_name.map(lambda x: x.lower().replace(',', '').replace('.', '') if (type(x) == str) else x)

    df['job_info_work_city'] = df['job_info_work_city'].map(lambda x: x.lower() if type(x) == str else x)
    df['agent_city'] = df['agent_city'].map(lambda x: x.lower() if type(x) == str else x)
    df['employer_name'] = df['employer_name'].map(lambda x: x.lower() if type(x) == str else x)
    df['foreign_worker_info_city'] = df['foreign_worker_info_city'].map(lambda x: x.lower() if type(x) == str else x)
    df['preparer_info_title'] = df['preparer_info_title'].map(lambda x: x.lower() if type(x) == str else x)
    df['fw_info_birth_country'] = df['fw_info_birth_country'].map(lambda x: x.lower() if type(x) == str else x)
    df['ri_1st_ad_newspaper_name'] = df['ri_1st_ad_newspaper_name'].map(lambda x: x.lower() if type(x) == str else x)
    df['ri_2nd_ad_newspaper_name'] = df['ri_2nd_ad_newspaper_name'].map(lambda x: x.lower() if type(x) == str else x)
    df['employer_city'] = df['employer_city'].map(lambda x: x.lower() if type(x) == str else x)
    df['foreign_worker_info_inst'] = df['foreign_worker_info_inst'].map(lambda x: x.lower() if type(x) == str else x)
    df['foreign_worker_info_major'] = df['foreign_worker_info_major'].map(lambda x: x.lower() if type(x) == str else x)
#     df['pw_job_title_908']
# pw_soc_title,
# pw_source_name_9089,
# preparer_info_emp_completed,
# employer_name,
# job_info_work_postal_code,
# fw_info_training_comp,
# fw_info_req_experience,
# ri_posted_notice_at_worksite,
# ri_2nd_ad_newspaper_or_journal,
# foreign_worker_info_state_abbr,
# recr_info_sunday_newspaper,
# num_employees_discrete,
# fw_info_rel_occup_exp,
# ji_foreign_worker_live_on_premises,
# job_info_training,
# ji_fw_live_on_premises,
# schd_a_sheepherder,
# agent_state_abbr
# employer_state_abbr

# TODO ft embedding
# job_info_job_title,
# pw_job_title_9089,
# naics_title,
# job_info_alt_occ_job_title,
# pw_job_title_908,
# job_info_major,

# numeric
# job_info_alt_cmb_ed_oth_yrs,
# job_info_alt_occ_num_months,
# job_info_experience_num_months,
# pw_amount_9089,

    print_attr_overview(df['case_status'], True, topn=10)

    s = df['employer_state']
    s = s.map(lambda x: x.lower() if type(x) == str else x)
    s = s.map(lambda x: us_state_abbrev[x] if (type(x) == str and (x.strip() in us_state_abbrev)) else x)
    s = s.map(lambda x: x.upper() if type(x) == str else x)
    df['employer_state_abbr'] = s

    s = df['agent_state']
    s = s.map(lambda x: x.lower() if type(x) == str else x)
    s = s.map(lambda x: us_state_abbrev[x] if (type(x) == str and (x.strip() in us_state_abbrev)) else x)
    s = s.map(lambda x: x.upper() if type(x) == str else x)
    df['agent_state_abbr'] = s

    s = df['job_info_work_state']
    s = s.map(lambda x: x.lower() if type(x) == str else x)
    s = s.map(lambda x: us_state_abbrev[x] if (type(x) == str and (x.strip() in us_state_abbrev)) else x)
    s = s.map(lambda x: x.upper() if type(x) == str else x)
    df['job_info_work_state_abbr'] = s
    
    s = df['foreign_worker_info_state']
    s = s.map(lambda x: x.lower() if type(x) == str else x)
    s = s.map(lambda x: us_state_abbrev[x] if (type(x) == str and (x.strip() in us_state_abbrev)) else x)
    s = s.map(lambda x: x.upper() if type(x) == str else x)
    df['foreign_worker_info_state_abbr'] = s

    # print(df.agent_state_abbr.value_counts())
    # print(df.employer_state_abbr.value_counts())
    # print(df.job_info_work_state_abbr.value_counts())
    # exit()

#     df['case_received_date_epoch'] = pd.to_datetime(df.case_received_date, errors='coerce').astype(np.int64) // 10**9
#     df['decision_date_epoch'] = pd.to_datetime(df.decision_date, errors='coerce').astype(np.int64) // 10**9
#     df['pw_determ_date_epoch'] = pd.to_datetime(df.pw_determ_date, errors='coerce').astype(np.int64) // 10**9
#     df['pw_expire_date_epoch'] = pd.to_datetime(df.pw_expire_date, errors='coerce').astype(np.int64) // 10**9

    date_cols = [s.replace('_epoch', '') for s in date_colnames]

    for feature_name in date_cols:
        print('preprocessing {}'.format(feature_name))
        df[feature_name + '_epoch'] = pd.to_datetime(df[feature_name], errors='coerce').astype(np.int64) // 10**9
    
    # colnames = ['naics_code']
    df['naics_code'][df['naics_us_code'].notnull()] = df['naics_us_code']

    logical_idx = df.naics_code.notnull()
    df['naics_sector'] = pd.Series()
    df['naics_sector'][logical_idx] = df.naics_code[logical_idx].astype(str).str[:2]
    print_attr_overview(df['naics_sector'], True)
    print_attr_overview(df['naics_code'], True, topn=25)

    print_attr_overview(df['class_of_admission'], True, topn=50)
    df['foreign_worker_info_birth_country'][np.logical_not(df['foreign_worker_info_birth_country'].notnull())] = df['country_of_citizenship']
    print_attr_overview(df['foreign_worker_info_birth_country'])
    # exit()

    s1 = df.foreign_worker_info_alt_edu_experience
    print_attr_overview(s1)
    s2 = df.fw_info_alt_edu_experience
    print_attr_overview(s2)

    # exit()

    s_combined = s1
    s1_is_null = np.logical_not(s1.notnull())
    s_combined[s1_is_null] = s2[s1_is_null]
    df['fw_info_alt_edu_experience'] = s_combined
    print_attr_overview(df['fw_info_alt_edu_experience'])
    # exit()

    # discretise:
    s = df.employer_num_employees
    df['num_employees_discrete'] = s.map(lambda x: discretise_employer_size(x) if (type(x) == float) else np.nan)
    print(df['employer_num_employees'])
    print(df['num_employees_discrete'])
    # exit()

    # Round to the nearest decade
    s = df.employer_yr_estab
    df['employer_yr_estab_rounded'] = s.map(lambda x: str(int(round(x, -1))) if (type(x) == float and not(np.isnan(x))) else "EMPTY_PLACEHOLDER")

    s = df.job_info_alt_occ_num_months
    df['job_info_alt_occ_num_months_str'] = s.map(lambda x: str(int(x)) if (type(x) == float and not(np.isnan(x))) else "EMPTY_PLACEHOLDER")

    s = df.job_info_experience_num_months
    df['job_info_experience_num_months_str'] = s.map(lambda x: str(int(x)) if (type(x) == float and not(np.isnan(x))) else "EMPTY_PLACEHOLDER")

    return df

def extract_features(df, selected_feature_names_categ, selected_feature_names_interval, shuffle=True, fuzzy_matching=True):

    features_to_use = []
    variable_types = []
    # exit()

    # Check to ensure all cols exist (avoid keyerrors)
    df[selected_feature_names_categ + selected_feature_names_interval]

    # for feature in selected_feature_names_categ:
    #     le = preprocessing.LabelEncoder()
    #     print(print_attr_overview(df[feature], True, topn=10))
    #     df[feature + '_encoded'] = le.fit_transform(df[feature])
    #     features_to_use.append(feature + '_encoded')
    
    le = preprocessing.LabelEncoder()
    all_unique = []

    for feature in selected_feature_names_categ:
        # print(print_attr_overview(df[feature], True, topn=10))
        
        s = df[feature]

        # Remove categorical entries with less than 10 occurances
        a = s.value_counts()
        s[s.isin(a.index[a < 12])] = np.nan

        s[s.isnull()] = "EMPTY_PLACEHOLDER"
        s = s.map(lambda x: x.lower() if type(x) == str else x)
        # print(np.unique(df[feature]))
        all_unique.extend(np.unique(s))

    # print(all_unique)
    le.fit(all_unique)
    # print(list(le.classes_))
    

    for feature in selected_feature_names_categ:
        # print(feature)
        # print(df[feature])
        s = df[feature]
        s = s.map(lambda x: x.lower() if type(x) == str else x)
        df[feature + '_encoded'] = le.transform(s)
        features_to_use.append(feature + '_encoded')
        variable_types.append('categorical_nominal')
        print(feature, len(np.unique(s)))
        # print()
    # exit()

    # Append interval AFTER categorical!!
    for feature in selected_feature_names_interval:
        s = df[feature]
        s = s.map(lambda x: x.replace(',', '') if type(x) == str else x)
        # print(s)
        s = pd.to_numeric(s, errors='coerce')
        s[np.logical_not(s.notnull())] = 0.0
        newname = feature + '_normed'
        
        # Normalize
        df[newname] = norm_zscore(s)
        features_to_use.append(newname)
        variable_types.append('numerical')

    X = df[features_to_use].as_matrix()
    y = df['case_status'].as_matrix()

    if shuffle:
        X, y = skl_shuffle(X, y)

    # print(X.shape)
    # print(y.shape)
    vocab_size = len(list(le.classes_))

    return X, y, vocab_size, variable_types, features_to_use

def split_categorical_and_interval(X, variable_types):
    idx_end_categorical = max(loc for loc, val in enumerate(variable_types) if val == 'categorical_nominal')

    # TODO replace with logical indexing?

    # Select only the categorical columns of X
    X_categorical = X[:, :idx_end_categorical+1]

    # Select only the numerical columns of X
    X_numerical = X[:, idx_end_categorical+1:]

    return X_categorical, X_numerical
