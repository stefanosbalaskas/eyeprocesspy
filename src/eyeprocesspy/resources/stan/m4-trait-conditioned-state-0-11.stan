data {
  int<lower=1> J;
  int<lower=1> I;
  int<lower=1> N;
  array[N] int<lower=1, upper=J> person;
  array[N] int<lower=1, upper=I> item;

  int<lower=1> N_response;
  array[N_response] int<lower=1, upper=N> response_row;
  array[N_response] int<lower=0, upper=1> y_response;

  array[N] int<lower=0, upper=1> rt_observed;
  vector[N] log_rt;
  array[N] int<lower=0, upper=1> gaze_observed;
  array[N] int<lower=0> gaze;
  array[N] int<lower=0, upper=1> pupil_observed;
  vector[N] pupil;

  matrix[N, 8] X_pupil;
  array[8] int<lower=0, upper=1> use_pupil_covariate;

  int<lower=1> S;
  array[S] int<lower=1, upper=N> seq_start;
  array[S] int<lower=1> seq_len;

  int<lower=1, upper=4> K;
  array[3] int<lower=0, upper=1> use_state_channel;
  int<lower=1, upper=2> transition_structure; // 1 = Markov, 2 = iid state membership
  array[4] int<lower=0, upper=1> use_transition_trait;
  array[4] int<lower=0, upper=1> use_initial_trait;
  int<lower=1, upper=2> prior_profile;
}

parameters {
  // Frozen M3 measurement backbone.
  matrix[4, J] z_person;
  vector<lower=0>[4] sigma_person;
  cholesky_factor_corr[4] L_person;

  matrix[4, I] z_item;
  vector[4] mu_item;
  vector<lower=0>[4] sigma_item;
  cholesky_factor_corr[4] L_item;

  vector<lower=0>[I] nu;
  vector<lower=0>[I] s;
  real<lower=0> sigma_pupil;
  vector[8] gamma_pupil;

  // State effects. Ordering on RT is an identification convention only.
  ordered[K] state_rt_raw;
  vector[K] state_gaze_raw;
  vector[K] state_pupil_raw;

  // Softmax logits are centered in transformed parameters so K = 1 is a
  // first-class null without zero-dimensional parameter blocks.
  vector[K] init_intercept_raw;
  matrix[4, K] init_trait_raw;
  matrix[K, K] trans_intercept_raw;
  array[4] matrix[K, K] trans_trait_raw;
}

transformed parameters {
  matrix[4, J] person_eff = diag_pre_multiply(sigma_person, L_person) * z_person;
  matrix[4, I] item_eff = rep_matrix(mu_item, I) + diag_pre_multiply(sigma_item, L_item) * z_item;

  vector[J] theta = (person_eff[1])';
  vector[J] tau = (person_eff[2])';
  vector[J] omega = (person_eff[3])';
  vector[J] rho = (person_eff[4])';

  vector[I] b = (item_eff[1])';
  vector[I] beta = (item_eff[2])';
  vector[I] m = (item_eff[3])';
  vector[I] kappa = (item_eff[4])';

  matrix[4, 4] corr_person = multiply_lower_tri_self_transpose(L_person);
  matrix[4, 4] corr_item = multiply_lower_tri_self_transpose(L_item);

  vector[K] delta_rt = state_rt_raw - mean(state_rt_raw);
  vector[K] delta_gaze = state_gaze_raw - mean(state_gaze_raw);
  vector[K] delta_pupil = state_pupil_raw - mean(state_pupil_raw);

  vector[K] init_intercept = init_intercept_raw - mean(init_intercept_raw);
  matrix[4, K] init_trait;
  matrix[K, K] trans_intercept;
  array[4] matrix[K, K] trans_trait;
  matrix[N, K] emission_lp;

  for (q in 1:4) {
    init_trait[q] = init_trait_raw[q] - mean(init_trait_raw[q]);
  }
  for (h in 1:K) {
    trans_intercept[h] = trans_intercept_raw[h] - mean(trans_intercept_raw[h]);
    for (q in 1:4) {
      trans_trait[q][h] = trans_trait_raw[q][h] - mean(trans_trait_raw[q][h]);
    }
  }

  for (n in 1:N) {
    real nuisance = 0;
    for (q in 1:8) {
      nuisance += use_pupil_covariate[q] * gamma_pupil[q] * X_pupil[n, q];
    }
    for (k in 1:K) {
      real lp = 0;
      if (rt_observed[n] == 1) {
        lp += normal_lpdf(
          log_rt[n] |
          beta[item[n]] - tau[person[n]] + use_state_channel[1] * delta_rt[k],
          1 / nu[item[n]]
        );
      }
      if (gaze_observed[n] == 1) {
        lp += neg_binomial_2_log_lpmf(
          gaze[n] |
          m[item[n]] + omega[person[n]] + use_state_channel[2] * delta_gaze[k],
          s[item[n]]
        );
      }
      if (pupil_observed[n] == 1) {
        lp += normal_lpdf(
          pupil[n] |
          kappa[item[n]] + rho[person[n]] + nuisance + use_state_channel[3] * delta_pupil[k],
          sigma_pupil
        );
      }
      emission_lp[n, k] = lp;
    }
  }
}

model {
  to_vector(z_person) ~ std_normal();
  to_vector(z_item) ~ std_normal();

  if (prior_profile == 1) {
    sigma_person ~ normal(0, 1);
    sigma_item ~ normal(0, 1);
    L_person ~ lkj_corr_cholesky(2);
    L_item ~ lkj_corr_cholesky(2);
    mu_item[1] ~ normal(0, 1);
    mu_item[2] ~ normal(4, 1);
    mu_item[3] ~ normal(3.5, 1.5);
    mu_item[4] ~ normal(0, 0.75);
    nu ~ lognormal(log(1.5), 0.5);
    s ~ lognormal(log(6), 0.7);
    sigma_pupil ~ normal(0, 0.75);
    gamma_pupil ~ normal(0, 0.5);

    // Identification anchor: separate the likelihood-irrelevant raw
    // location from adjacent ordered RT-state separations. The
    // Gamma gap prior has zero density at the singular zero-gap
    // boundary while retaining continuous support for weak separation.
    mean(state_rt_raw) ~ normal(0, 0.45);
    if (K > 1) {
      for (k in 2:K) {
        target += gamma_lpdf(
          state_rt_raw[k] - state_rt_raw[k - 1] | 3, 5
        );
      }
    }
    state_gaze_raw ~ normal(0, 0.45);
    state_pupil_raw ~ normal(0, 0.45);
    init_intercept_raw ~ normal(0, 0.8);
    to_vector(init_trait_raw) ~ normal(0, 0.35);
    to_vector(trans_intercept_raw) ~ normal(0, 0.9);
    for (q in 1:4) to_vector(trans_trait_raw[q]) ~ normal(0, 0.30);
  } else {
    sigma_person ~ normal(0, 2);
    sigma_item ~ normal(0, 2);
    L_person ~ lkj_corr_cholesky(1);
    L_item ~ lkj_corr_cholesky(1);
    mu_item[1] ~ normal(0, 0.5);
    mu_item[2] ~ normal(4, 0.5);
    mu_item[3] ~ normal(3.5, 1);
    mu_item[4] ~ normal(0, 1);
    nu ~ gamma(1, 1);
    s ~ inv_gamma(1, 1);
    sigma_pupil ~ normal(0, 1);
    gamma_pupil ~ normal(0, 1);

    // Broader paper-centered anchor prior: retain the same
    // zero-gap exclusion while allowing wider RT-state separation.
    mean(state_rt_raw) ~ normal(0, 0.75);
    if (K > 1) {
      for (k in 2:K) {
        target += gamma_lpdf(
          state_rt_raw[k] - state_rt_raw[k - 1] | 2, 2.5
        );
      }
    }
    state_gaze_raw ~ normal(0, 0.75);
    state_pupil_raw ~ normal(0, 0.75);
    init_intercept_raw ~ normal(0, 1.25);
    to_vector(init_trait_raw) ~ normal(0, 0.6);
    to_vector(trans_intercept_raw) ~ normal(0, 1.25);
    for (q in 1:4) to_vector(trans_trait_raw[q]) ~ normal(0, 0.5);
  }

  // Scored response remains the frozen M3 Rasch equation and is not state-dependent.
  for (r in 1:N_response) {
    int n = response_row[r];
    y_response[r] ~ bernoulli_logit(theta[person[n]] - b[item[n]]);
  }

  // Marginalized latent-state likelihood.
  for (ss in 1:S) {
    int start = seq_start[ss];
    int L = seq_len[ss];
    int j = person[start];
    vector[K] init_eta;
    vector[K] init_lp;

    for (k in 1:K) {
      init_eta[k] = init_intercept[k];
      for (q in 1:4) {
        init_eta[k] += use_initial_trait[q] * init_trait[q, k] * person_eff[q, j];
      }
    }
    init_lp = log_softmax(init_eta);

    if (transition_structure == 2) {
      for (tt in 1:L) {
        int n = start + tt - 1;
        target += log_sum_exp(init_lp + to_vector(emission_lp[n]'));
      }
    } else {
      vector[K] log_alpha = init_lp + to_vector(emission_lp[start]');
      if (L > 1) {
        for (tt in 2:L) {
          int n = start + tt - 1;
          vector[K] next_alpha;
          for (k in 1:K) {
            vector[K] acc;
            for (h in 1:K) {
              vector[K] trans_eta;
              vector[K] trans_lp;
              for (kk in 1:K) {
                trans_eta[kk] = trans_intercept[h, kk];
                for (q in 1:4) {
                  trans_eta[kk] += use_transition_trait[q] * trans_trait[q][h, kk] * person_eff[q, j];
                }
              }
              trans_lp = log_softmax(trans_eta);
              acc[h] = log_alpha[h] + trans_lp[k];
            }
            next_alpha[k] = log_sum_exp(acc) + emission_lp[n, k];
          }
          log_alpha = next_alpha;
        }
      }
      target += log_sum_exp(log_alpha);
    }
  }
}

generated quantities {
  vector[N_response] log_lik_response;
  array[N_response] int y_rep;
  vector[S] log_lik_process_sequence;

  matrix[N, K] state_prob;
  array[N] int<lower=1, upper=K> state_map;
  vector[N] state_entropy;
  array[N] int<lower=1, upper=K> state_rep;

  matrix[J, K] initial_prob_person;
  array[J] matrix[K, K] transition_prob_person;

  vector[N] log_rt_rep;
  array[N] int<lower=0> gaze_rep;
  vector[N] pupil_rep;

  for (r in 1:N_response) {
    int n = response_row[r];
    real eta = theta[person[n]] - b[item[n]];
    log_lik_response[r] = bernoulli_logit_lpmf(y_response[r] | eta);
    y_rep[r] = bernoulli_logit_rng(eta);
  }

  for (j in 1:J) {
    vector[K] init_eta;
    for (k in 1:K) {
      init_eta[k] = init_intercept[k];
      for (q in 1:4) {
        init_eta[k] += use_initial_trait[q] * init_trait[q, k] * person_eff[q, j];
      }
    }
    initial_prob_person[j] = softmax(init_eta)';

    for (h in 1:K) {
      vector[K] trans_eta;
      for (k in 1:K) {
        trans_eta[k] = trans_intercept[h, k];
        for (q in 1:4) {
          trans_eta[k] += use_transition_trait[q] * trans_trait[q][h, k] * person_eff[q, j];
        }
      }
      if (transition_structure == 1) {
        transition_prob_person[j][h] = softmax(trans_eta)';
      } else {
        transition_prob_person[j][h] = initial_prob_person[j];
      }
    }
  }

  // Forward-backward smoothed probabilities and sequence log likelihoods.
  for (ss in 1:S) {
    int start = seq_start[ss];
    int L = seq_len[ss];
    int j = person[start];
    vector[K] init_lp = log(to_vector(initial_prob_person[j]'));

    if (transition_structure == 2) {
      real seq_lp = 0;
      for (tt in 1:L) {
        int n = start + tt - 1;
        vector[K] lp = init_lp + to_vector(emission_lp[n]');
        real norm = log_sum_exp(lp);
        state_prob[n] = softmax(lp)';
        seq_lp += norm;
      }
      log_lik_process_sequence[ss] = seq_lp;
    } else {
      matrix[L, K] alpha;
      matrix[L, K] beta_bw;
      vector[K] first = init_lp + to_vector(emission_lp[start]');
      alpha[1] = first';

      if (L > 1) {
        for (tt in 2:L) {
          int n = start + tt - 1;
          vector[K] next_alpha;
          for (k in 1:K) {
            vector[K] acc;
            for (h in 1:K) {
              acc[h] = alpha[tt - 1, h] + log(transition_prob_person[j][h, k]);
            }
            next_alpha[k] = log_sum_exp(acc) + emission_lp[n, k];
          }
          alpha[tt] = next_alpha';
        }
      }

      for (k in 1:K) beta_bw[L, k] = 0;
      if (L > 1) {
        for (rev in 1:(L - 1)) {
          int tt = L - rev;
          int n_next = start + tt;
          for (h in 1:K) {
            vector[K] acc;
            for (k in 1:K) {
              acc[k] = log(transition_prob_person[j][h, k]) + emission_lp[n_next, k] + beta_bw[tt + 1, k];
            }
            beta_bw[tt, h] = log_sum_exp(acc);
          }
        }
      }

      log_lik_process_sequence[ss] = log_sum_exp(to_vector(alpha[L]'));
      for (tt in 1:L) {
        int n = start + tt - 1;
        vector[K] lp = to_vector(alpha[tt]') + to_vector(beta_bw[tt]');
        state_prob[n] = softmax(lp)';
      }
    }
  }

  for (n in 1:N) {
    real e = 0;
    int best = 1;
    real best_p = state_prob[n, 1];
    for (k in 1:K) {
      if (state_prob[n, k] > best_p) {
        best = k;
        best_p = state_prob[n, k];
      }
      if (state_prob[n, k] > 0) e -= state_prob[n, k] * log(state_prob[n, k]);
    }
    state_map[n] = best;
    state_entropy[n] = e;
  }

  // Replicated latent trajectories and process measurements for PPC.
  for (ss in 1:S) {
    int start = seq_start[ss];
    int L = seq_len[ss];
    int j = person[start];
    state_rep[start] = categorical_rng(to_vector(initial_prob_person[j]'));
    if (L > 1) {
      for (tt in 2:L) {
        int n = start + tt - 1;
        if (transition_structure == 1) {
          state_rep[n] = categorical_rng(to_vector(transition_prob_person[j][state_rep[n - 1]]'));
        } else {
          state_rep[n] = categorical_rng(to_vector(initial_prob_person[j]'));
        }
      }
    }
  }

  for (n in 1:N) {
    int k = state_rep[n];
    real nuisance = 0;
    real rt_mu = beta[item[n]] - tau[person[n]] + use_state_channel[1] * delta_rt[k];
    real gaze_eta = m[item[n]] + omega[person[n]] + use_state_channel[2] * delta_gaze[k];
    real pupil_mu;
    for (q in 1:8) {
      nuisance += use_pupil_covariate[q] * gamma_pupil[q] * X_pupil[n, q];
    }
    pupil_mu = kappa[item[n]] + rho[person[n]] + nuisance + use_state_channel[3] * delta_pupil[k];
    log_rt_rep[n] = normal_rng(rt_mu, 1 / nu[item[n]]);
    gaze_rep[n] = neg_binomial_2_log_rng(gaze_eta, s[item[n]]);
    pupil_rep[n] = normal_rng(pupil_mu, sigma_pupil);
  }
}
