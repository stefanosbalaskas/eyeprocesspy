data {
  int<lower=1> N;
  int<lower=2> K;
  int<lower=1> F;
  int<lower=1> P;
  int<lower=1> J;
  matrix[N, F] X;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=1, upper=P> person;
  array[N] int<lower=1, upper=J> item;
  array[N, K] int<lower=0, upper=1> available;
  matrix[K, F] signature;
  real<lower=0> anchor_strength;
}
parameters {
  simplex[K] mixing;
  matrix[K, F] feature_mean;
  matrix<lower=0>[K, F] feature_sd;
  vector[K] response_intercept;
  vector[P] ability_raw;
  vector[J] difficulty_raw;
  real<lower=0> ability_sd;
  real<lower=0> difficulty_sd;
}
transformed parameters {
  vector[P] ability = ability_sd * (ability_raw - mean(ability_raw));
  vector[J] difficulty = difficulty_sd * (difficulty_raw - mean(difficulty_raw));
}
model {
  ability_raw ~ std_normal();
  difficulty_raw ~ std_normal();
  ability_sd ~ normal(0, 1);
  difficulty_sd ~ normal(0, 1);
  response_intercept ~ normal(0, 1.5);
  to_vector(feature_sd) ~ lognormal(-0.3, 0.5);
  for (k in 1:K) {
    for (f in 1:F) {
      feature_mean[k, f] ~ normal(signature[k, f], inv_sqrt(1 + anchor_strength));
    }
  }
  for (n in 1:N) {
    vector[K] component = rep_vector(negative_infinity(), K);
    for (k in 1:K) {
      if (available[n, k] == 1) {
        real process_lp = 0;
        for (f in 1:F) process_lp += normal_lpdf(X[n, f] | feature_mean[k, f], feature_sd[k, f]);
        component[k] = log(mixing[k]) + process_lp
          + bernoulli_logit_lpmf(y[n] | response_intercept[k] + ability[person[n]] - difficulty[item[n]]);
      }
    }
    target += log_sum_exp(component);
  }
}
generated quantities {
  matrix[N, K] posterior_probability;
  vector[N] log_lik;
  for (n in 1:N) {
    vector[K] component = rep_vector(negative_infinity(), K);
    for (k in 1:K) {
      if (available[n, k] == 1) {
        real process_lp = 0;
        for (f in 1:F) process_lp += normal_lpdf(X[n, f] | feature_mean[k, f], feature_sd[k, f]);
        component[k] = log(mixing[k]) + process_lp
          + bernoulli_logit_lpmf(y[n] | response_intercept[k] + ability[person[n]] - difficulty[item[n]]);
      }
    }
    log_lik[n] = log_sum_exp(component);
    posterior_probability[n] = to_row_vector(softmax(component));
  }
}
