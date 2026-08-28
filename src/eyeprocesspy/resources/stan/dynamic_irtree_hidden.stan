functions {
  real uncertain_emission_lpmf(int observed, vector emission_probability, real certainty) {
    int K = num_elements(emission_probability);
    real probability = certainty * emission_probability[observed];
    if (K > 1) {
      probability += (1 - certainty) * (1 - emission_probability[observed]) / (K - 1);
    }
    return log(fmax(probability, 1e-12));
  }
}
data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=2> H;
  int<lower=2> Kobs;
  int<lower=1> D;
  matrix[N, D] X;
  array[N] int<lower=1, upper=Kobs> observed;
  vector<lower=1e-6, upper=1>[N] observation_certainty;
  array[S] int<lower=1, upper=N> sequence_start;
  array[S] int<lower=1> sequence_length;
  array[H, H] int<lower=0, upper=1> allowed_hidden;
  matrix<lower=0>[H, Kobs] emission_prior;
}
parameters {
  simplex[H] initial_probability;
  matrix[H, H] transition_intercept_raw;
  array[H] matrix[H, D] transition_beta_raw;
  array[H] simplex[Kobs] emission;
}
transformed parameters {
  matrix[H, H] transition_intercept;
  array[H] matrix[H, D] transition_beta;
  for (source in 1:H) {
    transition_intercept[source] = transition_intercept_raw[source] - mean(transition_intercept_raw[source]);
    for (d in 1:D) {
      transition_beta[source][, d] = transition_beta_raw[source][, d] - mean(transition_beta_raw[source][, d]);
    }
  }
}
model {
  initial_probability ~ dirichlet(rep_vector(1.0, H));
  to_vector(transition_intercept_raw) ~ normal(0, 1);
  for (source in 1:H) to_vector(transition_beta_raw[source]) ~ normal(0, 0.5);
  for (h in 1:H) emission[h] ~ dirichlet(to_vector(1 + 20 * emission_prior[h]));

  for (s in 1:S) {
    int first = sequence_start[s];
    int last = first + sequence_length[s] - 1;
    vector[H] log_alpha;
    for (h in 1:H) {
      log_alpha[h] = log(initial_probability[h])
        + uncertain_emission_lpmf(observed[first] | to_vector(emission[h]), observation_certainty[first]);
    }
    if (last > first) {
      for (n in (first + 1):last) {
        vector[H] next_alpha;
        for (destination in 1:H) {
          vector[H] candidates;
          for (source in 1:H) {
            vector[H] eta = to_vector(transition_intercept[source])
              + transition_beta[source] * to_vector(X[n]);
            for (candidate in 1:H) if (allowed_hidden[source, candidate] == 0) eta[candidate] = -1e10;
            {
              vector[H] log_transition = log_softmax(eta);
              candidates[source] = log_alpha[source] + log_transition[destination];
            }
          }
          next_alpha[destination] = uncertain_emission_lpmf(
            observed[n] | to_vector(emission[destination]), observation_certainty[n]
          ) + log_sum_exp(candidates);
        }
        log_alpha = next_alpha;
      }
    }
    target += log_sum_exp(log_alpha);
  }
}
generated quantities {
  matrix[N, H] filtered_probability;
  vector[S] sequence_log_lik;
  array[N] int<lower=1, upper=H> hidden_rep;
  array[N] int<lower=1, upper=Kobs> observed_rep;

  for (s in 1:S) {
    int first = sequence_start[s];
    int last = first + sequence_length[s] - 1;
    vector[H] log_alpha;

    hidden_rep[first] = categorical_rng(initial_probability);
    observed_rep[first] = categorical_rng(to_vector(emission[hidden_rep[first]]));
    for (h in 1:H) {
      log_alpha[h] = log(initial_probability[h])
        + uncertain_emission_lpmf(observed[first] | to_vector(emission[h]), observation_certainty[first]);
    }
    filtered_probability[first] = to_row_vector(softmax(log_alpha));

    if (last > first) {
      for (n in (first + 1):last) {
        vector[H] next_alpha;
        vector[H] eta_rep = to_vector(transition_intercept[hidden_rep[n - 1]])
          + transition_beta[hidden_rep[n - 1]] * to_vector(X[n]);
        for (candidate in 1:H) if (allowed_hidden[hidden_rep[n - 1], candidate] == 0) eta_rep[candidate] = -1e10;
        hidden_rep[n] = categorical_logit_rng(eta_rep);
        observed_rep[n] = categorical_rng(to_vector(emission[hidden_rep[n]]));

        for (destination in 1:H) {
          vector[H] candidates;
          for (source in 1:H) {
            vector[H] eta = to_vector(transition_intercept[source])
              + transition_beta[source] * to_vector(X[n]);
            for (candidate in 1:H) if (allowed_hidden[source, candidate] == 0) eta[candidate] = -1e10;
            {
              vector[H] log_transition = log_softmax(eta);
              candidates[source] = log_alpha[source] + log_transition[destination];
            }
          }
          next_alpha[destination] = uncertain_emission_lpmf(
            observed[n] | to_vector(emission[destination]), observation_certainty[n]
          ) + log_sum_exp(candidates);
        }
        log_alpha = next_alpha;
        filtered_probability[n] = to_row_vector(softmax(log_alpha));
      }
    }
    sequence_log_lik[s] = log_sum_exp(log_alpha);
  }
}
