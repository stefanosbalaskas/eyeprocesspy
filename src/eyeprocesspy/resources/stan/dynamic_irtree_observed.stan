data {
  int<lower=1> N;
  int<lower=2> K;
  int<lower=1> D;
  int<lower=1> P;
  int<lower=1> J;
  matrix[N, D] X;
  array[N] int<lower=1, upper=K> y;
  array[N] int<lower=1, upper=P> person;
  array[N] int<lower=1, upper=J> item;
  array[N] int<lower=1, upper=K> from_state;
  array[N, K] int<lower=0, upper=1> allowed;
  vector<lower=0>[N] observation_weight;
  int<lower=0, upper=1> use_person_re;
  int<lower=0, upper=1> use_item_re;
}
parameters {
  matrix[K, D] beta_raw;
  matrix[P, K] person_raw;
  matrix[J, K] item_raw;
  vector<lower=0>[K] sigma_person;
  vector<lower=0>[K] sigma_item;
}
transformed parameters {
  matrix[K, D] beta;
  matrix[P, K] person_effect;
  matrix[J, K] item_effect;
  for (d in 1:D) beta[, d] = beta_raw[, d] - mean(beta_raw[, d]);
  for (p in 1:P) {
    row_vector[K] value = person_raw[p] .* to_row_vector(sigma_person);
    person_effect[p] = value - mean(value);
  }
  for (j in 1:J) {
    row_vector[K] value = item_raw[j] .* to_row_vector(sigma_item);
    item_effect[j] = value - mean(value);
  }
}
model {
  to_vector(beta_raw) ~ normal(0, 1);
  to_vector(person_raw) ~ std_normal();
  to_vector(item_raw) ~ std_normal();
  sigma_person ~ normal(0, 0.5);
  sigma_item ~ normal(0, 0.5);
  for (n in 1:N) {
    vector[K] eta = beta * to_vector(X[n]);
    if (use_person_re == 1) eta += to_vector(person_effect[person[n]]);
    if (use_item_re == 1) eta += to_vector(item_effect[item[n]]);
    for (k in 1:K) if (allowed[n, k] == 0) eta[k] = -1e10;
    target += observation_weight[n] * categorical_logit_lpmf(y[n] | eta);
  }
}
generated quantities {
  vector[N] log_lik;
  array[N] int<lower=1, upper=K> y_rep;
  matrix[N, K] destination_probability;
  for (n in 1:N) {
    vector[K] eta = beta * to_vector(X[n]);
    if (use_person_re == 1) eta += to_vector(person_effect[person[n]]);
    if (use_item_re == 1) eta += to_vector(item_effect[item[n]]);
    for (k in 1:K) if (allowed[n, k] == 0) eta[k] = -1e10;
    destination_probability[n] = to_row_vector(softmax(eta));
    log_lik[n] = categorical_logit_lpmf(y[n] | eta);
    y_rep[n] = categorical_logit_rng(eta);
  }
}
