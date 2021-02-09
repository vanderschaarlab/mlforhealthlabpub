setwd('country_feature/')
library(dplyr)


dat_pop = read.csv('Popular Indicators/pop_data2.csv', stringsAsFactors = FALSE, na.strings = '..', nrows=310093)

head(dat_pop)
names(dat_pop)[1] = 'metric'

names(dat_pop)[5:9] = paste0('YR', 2015:2019)


dat_met = read.csv('selected_feature_names.csv', stringsAsFactors = FALSE)


dat_pop = dat_pop %>%
    inner_join(dat_met, by='metric')


dat_pop = dat_pop %>%
    mutate(value = coalesce(YR2019, YR2018, YR2017, YR2016, YR2015))

sum(is.na(dat_pop$value))

dat_miss = dat_pop[is.na(dat_pop$value), ]

feat_missing = dat_miss %>%
    group_by(metric) %>%
    summarise(n=n()) %>%
    filter(n>130)

dat_pop = dat_pop %>% filter(!(metric %in% feat_missing$metric))


dat_cleaned = dat_pop %>%
    select(metric, Country.Name, Country.Code, value)


## health care

dat_heal = read.csv('Data_Extract_From_Health_Nutrition_and_Population_Statistics/health_data.csv', stringsAsFactors = FALSE, na.strings = '..', nrows=104636)

head(dat_heal)
names(dat_heal)[1] = 'metric'


names(dat_heal)[5:9] = paste0('YR', 2015:2019)

# write.csv(unique(dat_heal$metric), file='heal_feat_name.csv', row.names=FALSE)

dat_heal = dat_heal %>%
    filter(metric %in% c('Prevalence of overweight (% of adults)'))


dat_heal = dat_heal %>%
    mutate(value = coalesce(YR2019, YR2018, YR2017, YR2016, YR2015))

sum(is.na(dat_heal$value))

dat_miss = dat_heal[is.na(dat_heal$value), ]

feat_missing = dat_miss %>%
    group_by(metric) %>%
    summarise(n=n()) 


dat_cleaned_heal = dat_heal %>%
    select(metric, Country.Name, Country.Code, value)

dat_cleaned = rbind(dat_cleaned, dat_cleaned_heal)

write.csv(dat_cleaned, row.names = FALSE, file='country_feats.csv')
