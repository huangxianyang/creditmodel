# -*- coding: utf-8 -*-
import numpy as  np

all_feature = ['status_of_existing_checking_account','duration_in_month','credit_history','purpose','credit_amount','savings_account_and_bonds',
           'present_employment_since','installment_rate_in_percentage_of_disposable_income','personal_status_and_sex','other_debtors_or_guarantors',
            'present_residence_since','property','age_in_years','other_installment_plans','housing','number_of_existing_credits_at_this_bank','job',
               'number_of_people_being_liable_to_provide_maintenance_for','telephone','foreign_worker']

cat_col = ['purpose','personal_status_and_sex','other_debtors_or_guarantors','property','other_installment_plans','housing','telephone','foreign_worker']
num_col = [i for i in all_feature if i not in cat_col]
int_col = cat_col+['status_of_existing_checking_account','credit_history','savings_account_and_bonds','job']
# 数值化
sub_col = ['status_of_existing_checking_account','credit_history','savings_account_and_bonds','present_employment_since','job']
rep_dict = {
    'status_of_existing_checking_account':{4:0},
    'savings_account_and_bonds':{5:np.nan},
    'purpose':{'A124':np.nan}
}