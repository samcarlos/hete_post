library(grf)
scenario = 1
directory = '/Users/sweiss/src/hete_net/hete_dgp/created_data/'
run_model = function(directory, scenario){
  print(scenario)
  t = read.csv(paste0(directory,'scenario_',scenario,'_t.csv'))
  x = read.csv(paste0(directory,'scenario_',scenario,'_x.csv'))
  y = read.csv(paste0(directory,'scenario_',scenario,'_y.csv'))
  
  t_train = t[(1:15000),]
  x_train = x[(1:15000),]
  y_train = y[(1:15000),]
  
  t_test = t[-c(1:15000),]
  x_test = x[-c(1:15000),]
  y_test = y[-c(1:15000),]
  
  
  tau.forest = causal_forest(x_train, y_train, t_train, tune.parameters = TRUE)
  
  predictions = predict(tau.forest, x_test)
  predictions = predictions[1]
  write.csv(predictions, file = paste0('/Users/sweiss/src/hete_net/hete_dgp/predicted_data_hete_net/grf_R_preds_','scenario_',scenario,'_t.csv'))
}

lapply(0:7,function(scenario) run_model('/Users/sweiss/src/hete_net/hete_dgp/created_data/',scenario))
