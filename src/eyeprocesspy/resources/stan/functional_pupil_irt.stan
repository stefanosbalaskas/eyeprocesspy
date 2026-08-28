data {
  int<lower=1> N_trial;
  int<lower=1> N_sample;
  int<lower=2> P;
  int<lower=2> J;
  int<lower=2> B;
  array[N_trial] int<lower=0, upper=1> response;
  array[N_trial] int<lower=1, upper=P> trial_person;
  array[N_trial] int<lower=1, upper=J> trial_item;
  array[N_sample] int<lower=1, upper=N_trial> sample_trial;
  array[N_sample] int<lower=1, upper=P> sample_person;
  array[N_sample] int<lower=1, upper=J> sample_item;
  vector[N_sample] pupil;
  matrix[N_sample, B] basis;
  vector[N_sample] luminance;
  vector[N_sample] gaze_x;
  vector[N_sample] gaze_y;
  array[N_sample] int<lower=0, upper=N_sample> previous_index;
  int<lower=0, upper=1> use_ar1;
  int<lower=0, upper=1> use_person_pupil;
  int<lower=0, upper=1> use_item_pupil;
}
parameters {
  vector[P] theta_raw;
  vector[J] difficulty_raw;
  vector[J] log_discrimination;
  vector[B] mean_curve;
  vector[B] theta_loading;
  vector[B] response_loading;
  vector[P] pupil_person_raw;
  vector[J] pupil_item_raw;
  real<lower=0> sigma_pupil_person;
  real<lower=0> sigma_pupil_item;
  real beta_luminance;
  real beta_gaze_x;
  real beta_gaze_y;
  real<lower=-0.99, upper=0.99> rho;
  real<lower=0> sigma;
}
transformed parameters {
  vector[P] theta = (theta_raw - mean(theta_raw)) / sd(theta_raw);
  vector[J] difficulty = difficulty_raw - mean(difficulty_raw);
  vector[P] pupil_person = sigma_pupil_person * pupil_person_raw;
  vector[J] pupil_item = sigma_pupil_item * pupil_item_raw;
  vector[N_sample] mu;
  for (n in 1:N_sample) {
    int trial = sample_trial[n];
    mu[n] = dot_product(basis[n], mean_curve)
      + theta[sample_person[n]] * dot_product(basis[n], theta_loading)
      + response[trial] * dot_product(basis[n], response_loading)
      + beta_luminance * luminance[n]
      + beta_gaze_x * gaze_x[n]
      + beta_gaze_y * gaze_y[n];
    if (use_person_pupil == 1) mu[n] += pupil_person[sample_person[n]];
    if (use_item_pupil == 1) mu[n] += pupil_item[sample_item[n]];
  }
}
model {
  theta_raw ~ std_normal();
  difficulty_raw ~ normal(0, 1);
  log_discrimination ~ normal(0, 0.25);
  mean_curve ~ normal(0, 0.5);
  theta_loading ~ normal(0, 0.25);
  response_loading ~ normal(0, 0.25);
  pupil_person_raw ~ std_normal();
  pupil_item_raw ~ std_normal();
  sigma_pupil_person ~ normal(0, 0.5);
  sigma_pupil_item ~ normal(0, 0.5);
  beta_luminance ~ normal(0, 0.5);
  beta_gaze_x ~ normal(0, 0.5);
  beta_gaze_y ~ normal(0, 0.5);
  rho ~ normal(0, 0.4);
  sigma ~ normal(0, 0.5);
  for (t in 1:N_trial) {
    response[t] ~ bernoulli_logit(exp(log_discrimination[trial_item[t]]) *
      (theta[trial_person[t]] - difficulty[trial_item[t]]));
  }
  for (n in 1:N_sample) {
    if (use_ar1 == 1 && previous_index[n] > 0) {
      pupil[n] - mu[n] ~ normal(rho * (pupil[previous_index[n]] - mu[previous_index[n]]), sigma);
    } else {
      pupil[n] ~ normal(mu[n], sigma / sqrt(1 - square(rho) * use_ar1));
    }
  }
}
generated quantities {
  vector[N_trial] response_probability;
  vector[N_trial] response_log_lik;
  vector[N_sample] pupil_log_lik;
  array[N_trial] int<lower=0, upper=1> response_rep;
  vector[N_sample] pupil_rep;
  for (t in 1:N_trial) {
    real eta = exp(log_discrimination[trial_item[t]]) *
      (theta[trial_person[t]] - difficulty[trial_item[t]]);
    response_probability[t] = inv_logit(eta);
    response_log_lik[t] = bernoulli_logit_lpmf(response[t] | eta);
    response_rep[t] = bernoulli_logit_rng(eta);
  }
  for (n in 1:N_sample) {
    real conditional_mean = mu[n];
    real conditional_sd = sigma / sqrt(1 - square(rho) * use_ar1);
    if (use_ar1 == 1 && previous_index[n] > 0) {
      conditional_mean += rho * (pupil[previous_index[n]] - mu[previous_index[n]]);
      conditional_sd = sigma;
    }
    pupil_log_lik[n] = normal_lpdf(pupil[n] | conditional_mean, conditional_sd);
    pupil_rep[n] = normal_rng(conditional_mean, conditional_sd);
  }
}
