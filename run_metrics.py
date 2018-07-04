import numpy as np
import pandas as pd

def load_data(scenario, directory = '/Users/sweiss/src/hete_net/hete_dgp/created_data/'):
  t = np.array(pd.read_csv(directory+'scenario_'+str(scenario)+'_t.csv'))
  t_x = np.array(pd.read_csv(directory+'scenario_'+str(scenario)+'_t_x.csv'))
  x = np.array(pd.read_csv(directory+'scenario_'+str(scenario)+'_x.csv'))
  y = np.array(pd.read_csv(directory+'scenario_'+str(scenario)+'_y.csv'))
  u_x = np.array(pd.read_csv(directory+'scenario_'+str(scenario)+'_y_mean.csv'))


  t_train = t[:15000]
  t_x_train = t_x[:15000]
  x_train = x[:15000]
  y_train = y[:15000]
  u_x_train = u_x[:15000]


  t_test = t[15000:]
  t_x_test = t_x[15000:]
  x_test = x[15000:]
  y_test = y[15000:]
  u_x_test  = u_x[15000:]

  return([t_train, t_test, t_x_train, t_x_test, x_train, x_test, y_train,y_test, u_x_train, u_x_test])


def q_score(hete_score, y_test, tmt_test):
    y = y_test.copy()
    d = hete_score.copy()
    ATE = y_test[tmt_test == 1].mean() - y_test[tmt_test == 0].mean()

    decreasing = np.argsort(d)[::-1]
    y_decreasing = y[decreasing]
    tmt_decreasing = tmt_test[decreasing]
    control_decreasing = 1 - tmt_decreasing
    y_tmt_decreasing = y_decreasing.copy()
    y_tmt_decreasing[control_decreasing == 1] = 0
    y_control_decreasing = y_decreasing.copy()
    y_control_decreasing[tmt_decreasing == 1] = 0

    lhs = y_tmt_decreasing.cumsum() / tmt_decreasing.cumsum()
    rhs = y_control_decreasing.cumsum() / control_decreasing.cumsum()
    N = decreasing.shape[0]
    random_policy = ATE * np.arange(1, N + 1) / N
    optimal_policy = lhs - rhs

    optimal_policy[optimal_policy == -np.inf] = 0
    optimal_policy[optimal_policy == np.inf] = 0
    optimal_policy[np.isnan(optimal_policy)] = 0
    out = optimal_policy - random_policy
    q = np.trapz(out, dx=1/N)
    return q


def sigmoid(x):
    return(1/(1+np.exp(-x)))


def true_profit(base_counterfactual, estimated_counterfactual, u_x):
    base_counterfactual = base_counterfactual.reshape(len(base_counterfactual), 1)
    estimated_counterfactual = estimated_counterfactual.reshape(len(estimated_counterfactual),1)
    u_x = u_x.reshape(len(u_x),1)

    gains = u_x.copy()
    gains[np.where(estimated_counterfactual > 0)[0]] = gains[np.where(estimated_counterfactual > 0)[0]] + .5*base_counterfactual[np.where(estimated_counterfactual > 0)[0]]
    gains = sigmoid(gains)

    return(np.sum(gains))

def expected_profit(y, estimated_counterfactual, t):

    y = y.reshape(len(y), 1)
    estimated_counterfactual = estimated_counterfactual.reshape(len(estimated_counterfactual),1)
    t = t.reshape(len(t),1)


    decision = (estimated_counterfactual>0)*1
    loc_decision_equals_t = np.where(decision == t)[0]

    y_hypothetical = y[loc_decision_equals_t]
    t_hypothetical = t[loc_decision_equals_t]

    response_t_1 = y_hypothetical[np.where(t_hypothetical == 1)[0]]
    response_t_0 = y_hypothetical[np.where(t_hypothetical == 0)[0]]


    return((response_t_1.sum() + response_t_0.sum() ) / len(t_hypothetical))



def load_return_score(scenario_number):
    t_train, t_test, t_x_train, t_x_test, X_train, X_test, y_train,y_test, u_x_train, u_x_test = load_data(scenario_number, directory = 'created_data/')
    hete_preds = pd.read_csv('/Users/sweiss/src/hete_net/hete_dgp/predicted_data_hete_net/hete_preds_'+str(scenario_number)+'.csv')
    hete_optim = pd.read_csv('/Users/sweiss/src/hete_net/hete_dgp/predicted_data_hete_net/hete_preds_optim_'+str(scenario_number)+'.csv')
    hete_r = pd.read_csv('/Users/sweiss/src/hete_net/hete_dgp/predicted_data_hete_net/hete_R_preds_scenario_'+str(scenario_number)+'_t.csv')
    preds = [hete_preds.iloc[:,1], np.log(hete_optim.iloc[:,2]/hete_optim.iloc[:,1]), hete_r.iloc[:,1], t_x_test]


    pct_accuracy_correct = [np.mean( ((t_x_test).reshape(len(x),1)>0) == (x.reshape(len(x),1)>0)) for x in preds]
    true_profits = [true_profit(t_x_test, x,u_x_test ) for x in preds]

    expected_profits = [expected_profit(y_test, x, t_test) for x in preds]
    q_scores = [q_score(x, y_test, t_test) for x in preds]


    return([q_scores,pct_accuracy_correct,true_profits,expected_profits])

scores = [load_return_score(x) for x in range(8)]

colnames = ['hete_net','hete_optim','hete_R','truth']
metrics = ['q_scores','pct_accuracy_correct','true_profits','expected_profits']

def save_scores(temp_metrics, metric, colnames):
    temp_metrics.columns = colnames
    temp_metrics.to_csv('/Users/sweiss/src/hete_net/hete_dgp/metric_results/'+metric+'.csv')

[save_scores(pd.DataFrame([q[x] for q in scores]), metrics[x], colnames) for x in range(4)]
