
import pickle
from utils.processing import *
from data.DELVE_data import *

from model.base_model import *
from model.R0forecast import * 


npi_vars       = ["npi_workplace_closing", "npi_school_closing", "npi_cancel_public_events",  
				  "npi_gatherings_restrictions", "npi_close_public_transport", "npi_stay_at_home", 
                  "npi_internal_movement_restrictions", "npi_international_travel_controls", "npi_masks"]

meta_features  = ["stats_population_density", "stats_median_age", "stats_gdp_per_capita", 
                  "stats_smoking", "stats_population_urban", "stats_population_school_age"]


SEIR_modeldate = "2020-09-14\\" 


def train_R0forecasting_model():

	country_list   = ["United States", "United Kingdom", "Italy", "Germany", "Brazil", "Japan", "Egypt"]

	projPATH       = SEIR_modeldate + "projections\\" 
	modPATH        = SEIR_modeldate + "models\\" 
	projections    = dict.fromkeys(country_list)
	models         = dict.fromkeys(country_list)

	for country in country_list: 

		projections[country] = pickle.load(open(projPATH + country, "rb"))
		models[country]      = pickle.load(open(modPATH + country, "rb"))


	country_dict   = get_COVID_DELVE_data(country_list)	


	Y              = []
	X_whether      = []
	X_metas        = [] 
	X_NPIs         = []
	X_mobility     = []
	X_stringency   = [] 

	for country in country_list: 
		X_whether.append(get_country_features(country_dict[country])[0])
		X_metas.append(get_country_features(country_dict[country])[1])
		X_mobility.append(get_country_features(country_dict[country])[2])
		X_NPIs.append(get_country_features(country_dict[country])[3])
		X_stringency.append(get_country_features(country_dict[country])[4])

		Y.append(get_beta(projections[country][3], models[country])[:X_whether[-1].shape[0]])
		Y[-1][:np.argmax(Y[-1])] = np.max(Y[-1]) 


	npi_model      = R0Forecaster(MAX_STEPS=len(Y[0]) + 150, BATCH_SIZE=len(Y), INPUT_SIZE=21, HIDDEN_UNITS=100, 
									NUM_LAYERS=2, EPOCH=5, LR=0.01, N_STEPS=100, alpha=0.05, mode="LSTM",
									country_parameters=country_dict, country_models=models)	

	npi_model.fit(X_whether, X_metas, X_mobility, X_NPIs, X_stringency, Y)

	file_npi_model  = open('PIPmodels\\R0Forecaster', 'wb')

	pickle.dump(npi_model, file_npi_model)


if __name__ == '__main__':

	train_R0forecasting_model()




