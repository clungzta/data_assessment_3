## first removal
### second removal

selected_feature_names_categ = [
    'country_of_citizenship', #FIXME
    ###'foreign_worker_info_education',
    'employer_state_abbr',
    'naics_sector',
    # 'naics_code',
    'class_of_admission',
    ###'foreign_worker_info_birth_country',
    'fw_info_birth_country',
    'fw_info_alt_edu_experience',
    ###'num_employees_discrete',
    ###'agent_state_abbr',
    # 'agent_firm_name',
    'agent_firm_name_modified',
    ###'employer_city',
    ### 'employer_decl_info_title',
    'employer_name',
    ###'employer_yr_estab_rounded',
    'job_info_alt_combo_ed',
    ###'foreign_worker_info_inst',
    ###'foreign_worker_info_city',
    ###'foreign_worker_info_major',
    'foreign_worker_info_training_comp',
    'fw_ownership_interest',
    'foreign_worker_ownership_interest',
    # 'foreign_worker_yr_rel_edu_completed_rounded',
    ###'job_info_alt_combo_ed_exp',
    'job_info_alt_field',
    'job_info_alt_occ_num_months_str',
    'job_info_experience_num_months_str',
##     'job_info_experience',
    'job_info_foreign_ed',
    'job_info_work_state_abbr',
    'job_info_foreign_lang_req',
    'job_info_job_req_normal',
    'pw_unit_of_pay_9089',
    'rec_info_barg_rep_notified',
    'recr_info_barg_rep_notified',
    'recr_info_professional_occ',
    'foreign_worker_info_rel_occup_exp',
    ###'foreign_worker_info_req_experience',
    ###'job_info_work_state',
##     'pw_level_9089',
    # wage_offer_from_9089
    # wage_offer_to_9089
    ###'wage_offer_unit_of_pay_9089',
    # pw_soc_title,
    'pw_source_name_9089',
    ###'preparer_info_emp_completed',
    ###'employer_name',
    ###'job_info_work_postal_code',
    ###'fw_info_training_comp',
    'fw_info_req_experience',
    'ri_posted_notice_at_worksite',
    'ri_2nd_ad_newspaper_or_journal',
    'foreign_worker_info_state_abbr',
    ###'recr_info_sunday_newspaper',
    'num_employees_discrete',
    'fw_info_rel_occup_exp',
    'ji_foreign_worker_live_on_premises',
    'job_info_training',
    'ji_fw_live_on_premises',
    'schd_a_sheepherder',
    ###'agent_state_abbr',
    'employer_state_abbr',
    # 'ri_1st_ad_newspaper_name',
    # 'ri_2nd_ad_newspaper_name'
]

date_colnames = [colname + '_epoch' for colname in [
    'case_received_date',
    'decision_date',
    'pw_expire_date',
    ###'pw_determ_date',    
    ###'ri_campus_placement_to',
    'recr_info_job_fair_from',
    ### 'ri_pvt_employment_firm_from',
    'ri_local_ethnic_paper_to',
    'ri_job_search_website_from',
    ###'recr_info_pro_org_advert_to',
    'recr_info_swa_job_order_end',
    ### 'ri_employee_referral_prog_to',
    ### 'ri_pvt_employment_firm_to',
    ### 'recr_info_swa_job_order_start',
    'ri_job_search_website_to',
    'recr_info_prof_org_advert_to',
    'recr_info_first_ad_start',
    'ri_employee_referral_prog_from',
    ###'ri_local_ethnic_paper_from',
    ###'ri_employer_web_post_to',
    ###'ri_campus_placement_from',
    # 'recr_info_radio_tv_ad_from',
    # 'recr_info_prof_org_advert_from',
    'recr_info_second_ad_start',
    # 'pw_expire_date'
     ]]

selected_feature_names_interval = [
##         'wage_offer_from_9089',   
###        'naics_code',
        'employer_num_employees',
##         'employer_yr_estab',
        'job_info_alt_cmb_ed_oth_yrs',
        'job_info_alt_occ_num_months',
        ### 'job_info_experience_num_months',
        ###'pw_amount_9089',
] + date_colnames

if __name__ == "__main__":
    from load_dataset import load_and_preprocess

    df = load_and_preprocess('TrainingSet(3).csv')
    used_features = selected_feature_names_categ + selected_feature_names_interval

    for colname in list(set(df.keys()) - set(used_features)):
        print(df[colname])
