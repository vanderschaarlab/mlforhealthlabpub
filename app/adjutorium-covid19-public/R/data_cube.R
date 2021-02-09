library(ggplot2)
library(dplyr)
library(reshape2)
library(ggforce)

dat = read.csv("data/CHESS COVID19 CaseReport 20200330.csv", stringsAsFactors = FALSE)
update_date = '2020-03-30'


d_patient = dat %>%
    # mutate_all(.funs=function(x) x = ifelse((x == 'Unknown') | (x == 'No') | (x == 'Borderline'), '', x)) %>%
    mutate(dateupdated = ifelse(nchar(dateupdated) == 0, update_date, dateupdated)) %>%
    mutate(dateupdated = substr(dateupdated, 1, 10)) %>%
    mutate_at(.vars=c('dateupdated', 'hospitaladmissiondate', 'dateadmittedicu', 'dateleavingicu', 'finaloutcomedate'),
              .funs = function(x) as.Date(x, format="%Y-%m-%d")) %>%
    mutate(hospital_ttc = as.numeric(dateupdated - hospitaladmissiondate),
           icu_ttc = as.numeric(dateupdated - dateadmittedicu),
           outcome_time = as.numeric(finaloutcomedate - hospitaladmissiondate),
           time_to_icu = as.numeric(dateadmittedicu - hospitaladmissiondate),
           icu_duration = as.numeric(dateleavingicu - dateadmittedicu)
    ) %>%
    mutate(finaloutcome = ifelse(finaloutcome == "Transfered", "", finaloutcome)) %>%
    mutate(sex = ifelse(sex == 'Male', 'Male', 'Female')) %>%
    mutate(
        is_2d_mortality = (outcome_time < 2 & finaloutcome == 'Death'),
        is_7d_mortality = (outcome_time < 7 & finaloutcome == 'Death'),
        is_2d_icu = time_to_icu < 2,
        is_7d_icu = time_to_icu < 7,
        n_comorbidity = (chonicrepiratory == 'Yes') +
            (asthmarequiring == 'Yes') +
            (hypertension == 'Yes') +
            (chronicheart == 'Yes') +
            (chronicrenal == 'Yes') +
            (chronicliver == 'Yes') +
            (isdiabetes == 'Yes') +
            (immunosuppressiondisease == 'Yes') +
            (obesityclinical == 'Yes') +
            (pregnancy == 'Yes')
    ) %>%
    select(
        caseid, 
        trustname, 
        sex,
        ageyear,
        n_comorbidity,
        chonicrepiratory,
        asthmarequiring,
        chronicheart,
        chronicrenal,
        chronicliver,
        isdiabetes,
        immunosuppressiondisease,
        immunosuppressiontreatment,
        obesityclinical,
        hypertension,
        pregnancy,
        finaloutcome,
        hospital_ttc,
        icu_ttc,
        outcome_time,
        time_to_icu,
        icu_duration,
        is_2d_mortality,
        is_7d_mortality,
        is_2d_icu,
        is_7d_icu,
        dateupdated,
        hospitaladmissiondate,
        dateadmittedicu,
        dateleavingicu,
        finaloutcomedate
    )

d_trust = d_patient %>%
    filter(!is.na(hospitaladmissiondate)) %>%
    filter(hospitaladmissiondate >= as.Date('2020-03-10', format="%Y-%m-%d")) %>%
    filter(hospitaladmissiondate < as.Date(update_date, format="%Y-%m-%d")) %>%
    group_by(trustname, hospitaladmissiondate) %>%
    summarise(
        newly_admitted = n(),
        n_male = sum(sex == 'Male'),
        n_female = sum(sex == 'Female'),
        comorbidity_gt1 = sum(n_comorbidity > 0),
        c0 = sum(n_comorbidity == 0),
        c1 = sum(n_comorbidity == 1),
        c2 = sum(n_comorbidity == 2),
        c3 = sum(n_comorbidity == 3),
        c4 = sum(n_comorbidity == 4),
        c_gt4 = sum(n_comorbidity > 4),
        chonicrepiratory = sum(chonicrepiratory == 'Yes'),
        asthmarequiring = sum(asthmarequiring == 'Yes'),
        chronicheart = sum(chronicheart == 'Yes'),
        chronicrenal = sum(chronicrenal == 'Yes'),
        chronicliver = sum(chronicliver == 'Yes'),
        isdiabetes = sum(isdiabetes == 'Yes'),
        hypertension = sum(hypertension == 'Yes'),
        immunosuppressiondisease = sum(immunosuppressiondisease == 'Yes'),
        immunosuppressiontreatment = sum(immunosuppressiontreatment == 'Yes'),
        obesityclinical = sum(obesityclinical == 'Yes'),
        pregnancy = sum(pregnancy == 'Yes'),
        is_2d_mortality = sum(is_2d_mortality, na.rm=TRUE),
        is_7d_mortality = sum(is_7d_mortality, na.rm=TRUE),
        is_2d_icu = sum(is_2d_icu, na.rm=TRUE),
        is_7d_icu = sum(is_7d_icu, na.rm=TRUE)
    )

d_total = d_trust %>%
    ungroup() %>%
    mutate(trustname = 'NATIONAL') %>%
    group_by(trustname, hospitaladmissiondate) %>%
    summarise_all(.funs=sum)

d_trust = rbind(d_trust, d_total)

d_patient_age = d_patient %>%
    select(trustname, ageyear)

write.csv(d_trust, file = '../assets/d_trust.csv', row.names = FALSE)

write.csv(d_patient_age, file = '../assets/d_patient_age.csv', row.names = FALSE)

d_date = d_patient %>%
    filter(hospitaladmissiondate >= as.Date('2020-03-10', format="%Y-%m-%d")) %>%
    filter(hospitaladmissiondate <= as.Date(update_date, format="%Y-%m-%d")) %>%
    select(dt = hospitaladmissiondate) %>%
    mutate(key=1)%>%
    distinct()

stay_breaks = c(0, 1, 8, 15, 999)
stay_icu_labels = c('New ICU Admission', 'Admitted ICU 1-7 Days Ago', 'Admitted ICU 8-14 Days Ago', 'Admitted ICU >14 Days Ago')

stay_labels = c('New Admission', 'Admitted 1-7 Days Ago', 'Admitted 8-14 Days Ago', 'Admitted >14 Days Ago')

d_stay = d_patient %>%
    filter(hospitaladmissiondate >= as.Date('2020-03-10', format="%Y-%m-%d")) %>%
    filter(hospitaladmissiondate <= as.Date(update_date, format="%Y-%m-%d")) %>%
    select(caseid, trustname, hospitaladmissiondate, finaloutcomedate) %>%
    mutate(key=1)%>%
    inner_join(d_date, by='key') %>%
    filter((hospitaladmissiondate <= dt)) %>%
    filter( is.na(finaloutcomedate) | (finaloutcomedate > dt)   ) %>%
    mutate(length_of_stay = as.numeric(dt - hospitaladmissiondate)) %>%
    mutate(length_of_stay_type = cut(length_of_stay, breaks = stay_breaks, right = FALSE, labels = stay_labels)) 


d_stay_trust = d_stay %>%
    group_by(trustname, dt, length_of_stay_type) %>%
    summarise(n = n())

d_stay_trust_total = d_stay_trust %>%
    ungroup() %>%
    mutate(trustname = 'NATIONAL') %>%
    group_by(trustname, dt, length_of_stay_type) %>%
    summarise(n = sum(n))

d_stay_trust = rbind(d_stay_trust_total, d_stay_trust)

write.csv(d_stay_trust, file = '../assets/d_stay_trust.csv', row.names = FALSE)

# ICU

d_icu_stay = d_patient %>%
    filter(dateadmittedicu >= as.Date('2020-03-10', format="%Y-%m-%d")) %>%
    filter(dateadmittedicu <= as.Date(update_date, format="%Y-%m-%d")) %>%
    select(caseid, trustname, dateadmittedicu, dateleavingicu) %>%
    mutate(key=1)%>%
    inner_join(d_date, by='key') %>%
    filter((dateadmittedicu <= dt)) %>%
    filter( is.na(dateleavingicu) | (dateleavingicu > dt)   ) %>%
    mutate(length_of_stay = as.numeric(dt - dateadmittedicu)) %>%
    mutate(length_of_stay_type = cut(length_of_stay, breaks = stay_breaks, right = FALSE, labels = stay_icu_labels)) 


d_icu_stay_trust = d_icu_stay %>%
    group_by(trustname, dt, length_of_stay_type) %>%
    summarise(n = n())

d_icu_stay_trust_total = d_icu_stay_trust %>%
    ungroup() %>%
    mutate(trustname = 'NATIONAL') %>%
    group_by(trustname, dt, length_of_stay_type) %>%
    summarise(n = sum(n))

d_icu_stay_trust = rbind(d_icu_stay_trust_total, d_icu_stay_trust)

write.csv(d_icu_stay_trust, file = '../assets/d_icu_stay_trust.csv', row.names = FALSE)

# admission, discharge, death

d_admission = d_patient %>%
    mutate(dt = hospitaladmissiondate) %>%
    filter(dt >= as.Date('2020-03-10', format="%Y-%m-%d")) %>%
    filter(dt <= as.Date(update_date, format="%Y-%m-%d")) %>%
    group_by(trustname, dt) %>%
    summarise(n_admission = n())

d_discharge = d_patient %>%
    mutate(dt = finaloutcomedate) %>%
    filter(finaloutcome == 'Discharged') %>%
    filter(dt >= as.Date('2020-03-10', format="%Y-%m-%d")) %>%
    filter(dt <= as.Date(update_date, format="%Y-%m-%d")) %>%
    group_by(trustname, dt) %>%
    summarise(n_discharge = n())

d_death = d_patient %>%
    mutate(dt = finaloutcomedate) %>%
    filter(finaloutcome == 'Death') %>%
    filter(dt >= as.Date('2020-03-10', format="%Y-%m-%d")) %>%
    filter(dt <= as.Date(update_date, format="%Y-%m-%d")) %>%
    group_by(trustname, dt) %>%
    summarise(n_death = n())

d_ts = d_admission %>%
    left_join(d_discharge, by=c('dt', 'trustname')) %>%
    left_join(d_death, by=c('dt', 'trustname'))

d_ts[is.na(d_ts)] = 0

d_total_ts = d_ts %>%
    ungroup() %>%
    mutate(trustname = 'NATIONAL') %>%
    group_by(trustname, dt) %>%
    summarise(n_admission = sum(n_admission),
              n_discharge = sum(n_discharge),
              n_death = sum(n_death))

d_ts = rbind(d_ts, d_total_ts)

write.csv(d_ts, file = '../assets/d_ts.csv', row.names = FALSE)

