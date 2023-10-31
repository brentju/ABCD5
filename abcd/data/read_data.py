"""Read subject and event data, combining several variables form the ABCD 5.0 release.
"""

import os
import pandas as pd
from tqdm import tqdm
import abcd.utils.io as io
from abcd.local.paths import core_path, output_path
import abcd.data.VARS as VARS

def get_subjects_events_visits(visits = ['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1']):
    name_suffix = '-'.join([v[0] for v in visits])
    try:
        subjects_df = io.load_df(output_path, "subjects_{}".format(name_suffix))
        events_df = io.load_df(output_path, "events_{}".format(name_suffix))
    except Exception:
        subjects_df, events_df = get_subjects_events_sf()
        subjects_df, events_df = filter_df_by_visits(subjects_df, events_df, visits = visits)
        # Adding per-subject NIH Neurocognition scores
        subjects_df = add_subject_vars(subjects_df, VARS.NIH_PATH, vars=list(VARS.NIH_TESTS_uncorrected.keys()) + list(VARS.NIH_COMBINED_uncorrected.keys()), leave_first=True)
        # Adding per-subject household income information
        subjects_df = add_subject_vars(subjects_df, VARS.DEMO_PATH, vars=["demo_comb_income_v2"], leave_first=True)
        subjects_df = subjects_df[subjects_df['demo_comb_income_v2'].isin(VARS.VALUES['demo_comb_income_v2'].keys())] # Filtering our 777 and 999
        events_df = filter_events(subjects_df, events_df)
        # Adding per-visit CBCL scores
        events_df = add_event_vars(events_df, VARS.CBCL_PATH, vars=list(VARS.CBCL_SCORES_t.keys())+list(VARS.CBCL_SCORES_raw.keys()))
        events_df = events_df.dropna() 
        subjects_df, events_df = filter_df_by_visits(subjects_df, events_df, visits=visits)
        # Adding sleeping habits
        events_df = add_event_vars(events_df, VARS.SLEEP_PATHS['sleepdisturb1_p'], vars=['sleepdisturb1_p', 'sleepdisturb2_p'])
        events_df = events_df.dropna() 
        subjects_df, events_df = filter_df_by_visits(subjects_df, events_df, visits=visits)
        io.dump_df(subjects_df, output_path, "subjects_{}".format(name_suffix))
        io.dump_df(events_df, output_path, "events_{}".format(name_suffix))
    return subjects_df, events_df

def get_subjects_events_sf():
    '''Fetches subjects and events with functional and structural features, as well as sex and 
    race/ethnicity information.
    '''
    try:
        subjects_df = io.load_df(output_path, "subjects_sfmri")
        events_df = io.load_df(output_path, "events_sfmri")
    except:
        subjects_df, events_df = get_subjects_events()
        print("There were initially {} visits for {} subjects".format(len(events_df), len(subjects_df)))
        # Add structural features
        for var_name, var_file in tqdm(VARS.DESIKAN_STRUCT_FILES.items()):
            file_path = os.path.join(core_path, "imaging", var_file)
            try:
                var_names = [var_name + '_' + x for x in VARS.DESIKAN_PARCELS[var_name] + VARS.DESIKAN_MEANS]
                events_df = add_event_vars(events_df, file_path, vars=var_names)
            except:
                var_names = [var_name + '_' + x for x in VARS.DESIKAN_PARCELS[var_name] + ['totallh', 'totalrh', 'total']]
                events_df = add_event_vars(events_df, file_path, vars=var_names)
                events_df = events_df.rename(columns=dict(zip([var_name + '_' + x for x in 
                    ['totallh', 'totalrh', 'total']], [var_name + '_' + x for x in VARS.DESIKAN_MEANS])))
            events_df = events_df.dropna()
            subjects_df = filter_subjects(subjects_df, events_df)
            print("After adding {}, there are {} visits for {} subjects".format(var_name, len(events_df), len(subjects_df)))
        io.dump_df(subjects_df, output_path, "subjects_sfmri")
        io.dump_df(events_df, output_path, "events_sfmri")
    return subjects_df, events_df
    
def get_subjects_events():
    '''Fetches subjects for which there are resting state fMRI connectivity scores, info. on the sex
    assigned at birth and who have only been scanned in one site.
    
    Returns:
        subjects_df (pandas.DataFrame): Subjects dataframe with site, family and sex info. 
        events_df (pandas.DataFrame): Events dataframe with RS fMRI connectivity info.
    '''
    try:
        subjects_df = io.load_df(output_path, "subjects_fmri")
        events_df = io.load_df(output_path, "events_fmri")
    except:
        subjects_df, events_df = read_events_general_info()
        subjects_df, events_df = add_event_connectivity_scores(subjects_df, events_df)
        subjects_df, events_df = add_subject_sex(subjects_df, events_df)
        subjects_df, events_df = add_subject_ethnicity(subjects_df, events_df)
        io.dump_df(subjects_df, output_path, "subjects_fmri")
        io.dump_df(events_df, output_path, "events_fmri")
    return subjects_df, events_df

def filter_df_by_visits(subjects_df, events_df, visits = ['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1']):
    '''Leave only subjects for which there is data for all the specified visits
    '''
    complete_subjects = []
    for subject_id in tqdm(set(events_df['src_subject_id'])):
        subject_events_df = events_df.loc[(events_df['src_subject_id'] == subject_id)]
        eventnames = set(subject_events_df['eventname'])
        if all([x in eventnames for x in visits]):
            complete_subjects.append(subject_id)
    complete_events_df = events_df[events_df['src_subject_id'].isin(complete_subjects)]
    complete_events_df = complete_events_df[complete_events_df['eventname'].isin(visits)]
    complete_subjects_df = filter_subjects(subjects_df, complete_events_df)
    assert len(complete_events_df) == len(visits)*len(complete_subjects_df)
    return complete_subjects_df, complete_events_df
    
def add_event_connectivity_scores(subjects_df, events_df):
    '''Add Resting state fMRI - Correlations (Gordon network). Filter subjects without connections.
    '''
    new_events_df = add_event_vars(events_df, VARS.fMRI_PATH, vars=list(VARS.NAMED_CONNECTIONS.keys()))
    new_events_df = add_event_vars(new_events_df, VARS.fMRI_to_subcortical_PATH, vars=list(VARS.CONNECTIONS_C_SC.keys()))
    new_events_df = new_events_df.dropna() 
    new_subjects_df = filter_subjects(subjects_df, new_events_df)
    return new_subjects_df, new_events_df

def add_subject_sex(subjects_df, events_df):
    '''Add the sex assigned at birth to each subject. Filter events of subjects without sex.
    These have the following values: 1, Male | 2, Female | 999, Don't know | 777, Refuse to answer
    '''
    table_path = os.path.join(core_path, "gender-identity-sexual-health", "gish_y_gi.csv")
    new_subjects_df = add_subject_vars(subjects_df, table_path, vars=["kbi_sex_assigned_at_birth"])
    # Remove subjects with "don't know" or "refuse to answer" sex
    new_subjects_df = new_subjects_df[new_subjects_df['kbi_sex_assigned_at_birth'].isin([1.0, 2.0])]
    new_events_df = filter_events(new_subjects_df, events_df)
    return new_subjects_df, new_events_df

def add_subject_ethnicity(subjects_df, events_df):
    '''Add the subject's race/ethnicity
    '''
    table_path = os.path.join(core_path, "abcd-general", "abcd_p_demo.csv")
    new_subjects_df = add_subject_vars(subjects_df, table_path, vars=["race_ethnicity"])
    new_subjects_df = new_subjects_df.dropna()
    new_events_df = filter_events(new_subjects_df, events_df)
    return new_subjects_df, new_events_df

def subject_cols_to_events(subjects_df, events_df, columns):
    events_df = pd.merge(events_df, subjects_df[['src_subject_id'] + columns], on='src_subject_id', how='left')
    return events_df

def event_cols_to_subjects(subjects_df, events_df, columns, visits = ['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1']):
    '''Add columns from the events dataframe into the subject one, succeding the column with the 
    visit name'''
    for visit in visits:
        visit_events_df = events_df[events_df['eventname'] == visit] 
        assert len(subjects_df) == len(visit_events_df)
        subjects_df = pd.merge(subjects_df, visit_events_df[['src_subject_id']+columns], on='src_subject_id')
        # Add visit name suffix
        subjects_df = subjects_df.rename(columns={col: col+'_'+visit for col in columns})
    return subjects_df

def filter_subjects(subjects_df, events_df):
    '''Filter subjects for which there are no events'''
    return subjects_df[subjects_df.apply(lambda x: 
        len(events_df[events_df['src_subject_id'] == x['src_subject_id']]) > 0, axis=1)]

def filter_events(subjects_df, events_df):
    '''Filter events for which there are no matching subjects'''
    subject_ids = list(subjects_df['src_subject_id'])
    return events_df[events_df.src_subject_id.isin(subject_ids)]

def add_subject_vars(subjects_df, table_path, vars=[], leave_first=False):
    '''Add additional information to the subjects dataframe. If there are more than one entries for
    that value, the subject is filtered out.
    
    Parameters:
        subjects_df (pandas.DataFrame): Subjects dataframe
        table_path (str): path to the additional table
        vars ([str]): list of variables to be added to that table
        leave_first (bool): If true and more than 1 value per subject, the first is left. If false, 
            ony subjects are left with exactly one value.
    Returns:
        subjects_df (pandas.DataFrame)
    '''
    df = io.load_df(table_path, sep =',')
    subject_ids = subjects_df['src_subject_id']
    new_df = []
    for subject_id in tqdm(subject_ids):
        subject_df = df[df['src_subject_id'] == subject_id] 
        new_row = [subject_id]
        for var in vars:
            values = subject_df[var].dropna().unique()
            if len(values) == 1:
                new_row.append(values[0])
            else:
                if leave_first and len(values) > 1: 
                    new_row.append(values[0])
                else:
                    new_row.append(None)
        if all(new_row):
            new_df.append(new_row)  # Filter out subjects with missing values (or >1 per subject)
        #else:
        #    print('Subject {} removed'.format(subject_id))
    new_df = pd.DataFrame(new_df, columns=['src_subject_id'] + vars)
    new_subjects_df = pd.merge(subjects_df, new_df, on=["src_subject_id"])
    return new_subjects_df
    
def add_event_vars(events_df, table_path, vars=[]):
    '''Add additional information to the events dataframe.
    
    Parameters:
        events_df (pandas.DataFrame): Events dataframe
        table_path (str): path to the additional table
        vars ([str]): list of variables to be added to that table
    Returns:
        events_df (pandas.DataFrame)
    '''
    new_df = io.load_df(table_path, sep =',', cols=["src_subject_id", "eventname"]+vars)
    new_df = new_df.groupby(["src_subject_id", "eventname"])[vars].mean()
    new_events_df = pd.merge(events_df, new_df, on=["src_subject_id", "eventname"])
    return new_events_df

def read_events_general_info(output_path=output_path):
    '''Read general info for all events.
    
    Parameters:
        output_path (str): Optional output path to store and restore dataframes.
    Returns:
        subjects_df (pandas.DataFrame): Subjects dataframe. Subjects are 
            removed which have been scanned in more than one site.
        events_df (pandas.DataFrame): Events dataframe. Events of removed subjects are removed.
    '''
    if output_path:
        try:
            subjects_df = io.load_df(output_path, "subjects_gi")
            events_df = io.load_df(output_path, "events_gi")
            return subjects_df, events_df
        except:
            print('Processing general information for the first time.')
    # File abcd_y_lt contains the interview date for each event, Site ID and Family ID, which is 
    # stored only for the baseline event.
    general_info_file = os.path.join(core_path, "abcd-general", "abcd_y_lt.csv")
    # Read all events from the main table
    df = io.load_df(general_info_file, sep =',')
    subjects, events = dict(), []
    filtered_subject_ids = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        subject_id, site_id  = row['src_subject_id'], row['site_id_l']
        if subject_id not in subjects and subject_id not in filtered_subject_ids:
            subjects[subject_id] = [subject_id, site_id, row['rel_family_id']]
        elif subject_id in filtered_subject_ids:
            continue
        elif site_id != subjects[subject_id][1]:
            filtered_subject_ids.append(subject_id)
            del subjects[subject_id]
            continue
        events.append([subject_id, row['interview_date'], row['eventname'], row['interview_age']])
    subjects = list(subjects.values())
    subjects_df = pd.DataFrame(subjects, columns=['src_subject_id', 'site_id_l', 'rel_family_id'])
    events = [event for event in events if event[0] not in filtered_subject_ids]
    events_df = pd.DataFrame(events, columns=['src_subject_id', 'interview_date', 'eventname', 'interview_age'])
    events_df = events_df.dropna() 
    subjects_df = filter_subjects(subjects_df, events_df)
    if output_path:
        io.dump_df(subjects_df, output_path, "subjects_gi")
        io.dump_df(events_df, output_path, "events_gi")
    return subjects_df, events_df



