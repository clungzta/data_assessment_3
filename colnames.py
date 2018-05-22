selected_feature_names_categ = [
    'country_of_citizenship',
    'foreign_worker_info_education',
    'employer_state_abbr',
    'naics_sector',
    # 'naics_code',
    'class_of_admission',
    'foreign_worker_info_birth_country',
    'fw_info_alt_edu_experience',
    'num_employees_discrete',
    'agent_state_abbr',
    'agent_firm_name',
    'agent_firm_name_modified',
    'employer_city',
    'employer_decl_info_title',
    'employer_name_modified',
    'employer_yr_estab_rounded',
    'job_info_alt_combo_ed',
    'foreign_worker_info_inst',
    'foreign_worker_info_major',
    'foreign_worker_info_training_comp',
    'fw_ownership_interest',
    'foreign_worker_ownership_interest',
    # 'foreign_worker_yr_rel_edu_completed_rounded',
    'job_info_alt_combo_ed_exp',
    'job_info_alt_field',
    'job_info_alt_occ_num_months_str',
    'job_info_experience_num_months_str',
    'job_info_experience',
    'job_info_foreign_ed',
    'job_info_work_state_abbr',
    'job_info_foreign_lang_req',
    'job_info_job_req_normal',
    'pw_unit_of_pay_9089',
    'rec_info_barg_rep_notified',
    'recr_info_barg_rep_notified',
    'recr_info_professional_occ',
    'foreign_worker_info_rel_occup_exp',
    'foreign_worker_info_req_experience',
    'job_info_work_state',
    'pw_level_9089',
    # wage_offer_from_9089
    # wage_offer_to_9089
    'wage_offer_unit_of_pay_9089'
]

selected_feature_names_interval = [
        'wage_offer_from_9089',
        'naics_code',
        'employer_num_employees',
        'employer_yr_estab',
        'case_received_date_epoch',
        'decision_date_epoch',
        'pw_expire_date_epoch',
        'pw_determ_date_epoch',
        'foreign_worker_yr_rel_edu_completed',
]

if __name__ == "__main__":
    from load_dataset import load_and_preprocess

    df = load_and_preprocess('TrainingSet(3).csv')    
    used_features = selected_feature_names_categ + selected_feature_names_interval

    for colname in list(set(df.keys()) - set(used_features)):
        print(df[colname])
    
