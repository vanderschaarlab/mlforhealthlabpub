
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from rpy2.robjects import r, pandas2ri
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
rpy2.robjects.numpy2ri.activate()

predict_features = ['AGE', 'ScreeningvsClinical', 'TUMOURSIZE','GRADEG1', 'GRADEG2', 'GRADEG3', 'GRADEG4', 
                    'NODESINVOLVED', 'ER_STATUSN', 'HER2_STATUSP']


r('''predict <- function(time, age, screen, size, grade, nodes,
                    er, her2, ki67, 
                    generation, horm, traz, bis) {

      # Grade variable for ER neg
      grade.val <- ifelse(er==1, grade, ifelse(grade==1, 0, 1)) 
  
      # Generate the coefficients
      
      age.mfp.1   <- ifelse(er==1, (age/10)^-2-.0287449295, age-56.3254902)
      age.beta.1  <- ifelse(er==1, 34.53642, 0.0089827)
      age.mfp.2   <- ifelse(er==1, (age/10)^-2*log(age/10)-.0510121013, 0)
      age.beta.2  <- ifelse(er==1, -34.20342, 0)
      size.mfp    <- ifelse(er==1, log(size/100)+1.545233938, (size/100)^.5-.5090456276)
      size.beta   <- ifelse(er==1, 0.7530729, 2.093446)
      nodes.mfp   <- ifelse(er==1,log((nodes+1)/10)+1.387566896,
                        log((nodes+1)/10)+1.086916249)
      nodes.beta  <- ifelse(er==1, 0.7060723, .6260541)
      grade.beta  <- ifelse(er==1, 0.746655, 1.129091)
      screen.beta <- ifelse(er==1, -0.22763366, 0)
      her2.beta   <- ifelse(her2==1, 0.2413,
                        ifelse(her2==0, -0.0762,0 ))
      ki67.beta   <- ifelse(ki67==1 & er==1, 0.14904,
                        ifelse(ki67==0 & er==1, -0.1133,0 ))
  
      # Other mortality prognostic index (mi)
      mi <- 0.0698252*((age/10)^2-34.23391957)
  
      # Breast cancer mortality prognostic index (pi)
      pi <- age.beta.1*age.mfp.1 + age.beta.2*age.mfp.2 + size.beta*size.mfp +
        nodes.beta*nodes.mfp + grade.beta*grade.val + screen.beta*screen + 
        her2.beta + ki67.beta
  
      # Treatment coefficients
      c     <- ifelse(generation == 0, 0, ifelse(generation == 2, -.248, -.446))
      h     <- ifelse(horm==1 & er==1, -0.3857, 0)
      t     <- ifelse(her2==1 & traz==1, -.3567, 0)
      b     <- ifelse(bis==1, -0.198, 0) # Only applicable to menopausal women.
  
      # Treatment combined
      rx <- c(0, h, h+c, h+c+t, h+c+t+b)
  
      # Non breast cancer mortality
      # Generate cumulative baseline other mortality
      base.m.cum.oth <- exp(-6.052919 + (1.079863*log(time)) + (.3255321*time^.5))
  
      # Generate cumulative survival non-breast mortality
      s.cum.oth <- exp(-exp(mi)*base.m.cum.oth)
  
      # Generate annual survival from cumulative survival
      m.cum.oth <- 1 - s.cum.oth 
  
  
      # Breast cancer specific mortality
      # Generate cumulative baseline breast mortality
      if (er==1) {
        base.m.cum.br <- exp(0.7424402 - 7.527762/time^.5 - 1.812513*log(time)/time^.5) 
      } else { base.m.cum.br <- exp(-1.156036 + 0.4707332/time^2 - 3.51355/time)
      }
  
      # Calculate the cumulative breast cancer survival
      s.cum.br <- exp(-exp(pi+rx)*base.m.cum.br)
      m.cum.br <- 1 - s.cum.br
  
      # All cause mortality
      m.cum.all <- 1 - s.cum.oth*s.cum.br
      s.cum.all <- 100-100*m.cum.all
  
  
      # Proportion of all cause mortality that is breast cancer
      prop.br <- m.cum.br/(m.cum.br+m.cum.oth)
      prop.oth <- m.cum.oth/(m.cum.br+m.cum.oth)
  
      # Predicted cumulative breast specific mortality
      pred.m.br    <- prop.br*m.cum.all
  
      # Predicted cumulative non-breast cancer mortality
      pred.m.oth <- prop.oth*m.cum.all
  
      # Predicted cumulative all-cause mortality
      pred.all <- pred.m.br #+ pred.m.oth
      intervention <- c("surgery", "h", "hc", "hct", "hctb")
      l <- list(intervention=intervention, surv = 100 - 100*pred.all)
      out <- l
      return(out)
      }''')
    


def PREDICT(Input_vec, horizon):
    
    horizon_val = str(horizon)
    age_val     = str(Input_vec['AGE'])
    screen_val  = str(Input_vec['ScreeningvsClinical'])
    tumour_val  = str(Input_vec['TUMOURSIZE'])
    grade_val   = str(Input_vec['GRADEG1']*1+ Input_vec['GRADEG2']*2+ Input_vec['GRADEG3']*3+ Input_vec['GRADEG4']*4)
    nodes_val   = str(Input_vec['NODESINVOLVED'])
    er_val      = str(np.floor(1-Input_vec['ER_STATUSN']))
    her_val     = str(np.floor(Input_vec['HER2_STATUSP']))

    r("temp <- predict(time=" + horizon_val + ", age=" + age_val +", screen=" + screen_val + ", size=" + tumour_val + ", grade=" + grade_val +", nodes=" + nodes_val + ",er=," + er_val + ", her2=" + her_val + ",ki67=0,generation=2, horm=0, traz=0, bis=0)")

    return r.temp[1][0], r.temp[1][2]
    

class PREDICT_model:
    
    def __init__(self, max_scaler):
        
        self.minmaxscaler     = max_scaler        
        self.predict_features = ['AGE', 'ScreeningvsClinical', 'TUMOURSIZE','GRADEG1', 'GRADEG2', 
                                 'GRADEG3', 'GRADEG4', 'NODESINVOLVED', 'ER_STATUSN', 'HER2_STATUSP']
        
    def predict_proba(self, x):
        
        x_prepro              = self.minmaxscaler.inverse_transform(x)
        self.features_dict    = dict.fromkeys(predict_features)
        
        for k in range(len(self.features_dict)):
    
            if x.shape[0]==1:
            
                self.features_dict[self.predict_features[k]] = x_prepro[:, k]
                
            else:
                
                self.features_dict[self.predict_features[k]] = x_prepro[:, k]
            
    
        x_pandas = pd.DataFrame.from_dict(self.features_dict) 
        preds1   = (100-np.array(x_pandas.apply(lambda x: PREDICT(Input_vec=x[self.predict_features], horizon=5)[0], axis=1)))/100
        preds    = np.hstack((1-preds1.reshape((-1,1)), preds1.reshape((-1,1))))
        
        return preds

