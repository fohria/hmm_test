"""
    example use:
    ```
    λ = setup_example()
    Q, O = generate_observations(λ, 3)
    # so total likelihood of model λ given observations..
    prob_of_obs_given_lambda(O, λ)
    ```
"""
import numpy as np
from itertools import product


def setup_example():
    """
        setup example model, no inputs. returns model λ
        TODO: consolidate this function with luis'
    """
    π = lambda: np.random.choice(['S1', 'S2', 'S3'], p=[1/3, 1/3, 1/3])  # 1/3 ok for now
    A = np.array([
        [0.4, 0.3, 0.3],
        [0.2, 0.6, 0.2],
        [0.1, 0.1, 0.8]
    ])  # = {a_ij}, the set of all a_ij
    # B = {  # in this case we have (see below) a markov model - special case of HMM
    #     'S1': 'red',  # np.random.choice(['red', 'green', 'blue']),
    #     'S2': 'green',
    #     'S3': 'blue'
    # }
    B = {'S1': {'red': 1, 'green': 0, 'blue': 0},
         'S2': {'red': 0, 'green': 1, 'blue': 0},
         'S3': {'red': 0, 'green': 0, 'blue': 1}
    }
    # B = {'S1': {'red': 0.9, 'green': 0.05, 'blue': 0.05},
    #      'S2': {'red': 0.05, 'green': 0.9, 'blue': 0.05},
    #      'S3': {'red': 0.05, 'green': 0.05, 'blue': 0.9}
    # }
    λ = {
        'A': A,
        'B': B,
        'π': π
    }

    return λ


def generate_observations(λ, timesteps):
    """
        generate observations given a model λ
        returns tuple of states, observations
        TODO: consolidate this function with luis'
    """
    A = λ['A']
    B = λ['B']
    π = λ['π']
    A_map = {
        'S1': 0,
        'S2': 1,
        'S3': 2
    }
    T = timesteps  # time steps
    Q = [π()]  # state sequence
    o_0 = np.random.choice(['red', 'green', 'blue'], p=list(B[Q[0]].values()))
    O = [o_0]  # ugh, but use first state in B to generate observation

    for t in range(1, T):
        q_t = np.random.choice(['S1', 'S2', 'S3'], p=A[A_map[Q[t-1]]])
        Q.append(q_t)
        o_t = np.random.choice(
            ['red', 'green', 'blue'], p=list(B[q_t].values()))
        O.append(o_t)

    return Q, O


def alpha_base_case(π, B, o_1):
    # initial state probs, obs prob function, observation at first timestep
    # pi is 1/3 for all states but keeping it as parameter for completeness
    alphas = [1/3 * b[o_1] for b in B.values()]
    # return sum(alphas)
    return np.array(alphas)


def alpha_inductive_step(A, B, o_t, α_previous):
    """
        transmatrix, obs prob function, observation at timestep
    """

    N = 3
    a = A
    b = [B[state] for state in B.keys()]
    a_j = []

    for j in range(N):  # switch N for some other later?
        asum = sum([α_previous[i] * a[i,j] for i in range(N)])
        a_j.append(asum * b[j][o_t])

    return np.array(a_j)


def alpha(O, λ):
    """
        output of this can be used like so:
        # P(O | λ) = Σα_T(i) for i=1 to N
        sum(alpha(O, λ)[len(O)-1])
    """
    N = len(λ['B'])  # nbr of states
    T = len(O)  # nbr of observations/timesteps
    α = np.zeros((T, N))
    α[0] = alpha_base_case(λ['π'], λ['B'], O[0])

    for i in range(1, T):
        α[i] = alpha_inductive_step(λ['A'], λ['B'], O[i], α[i-1])

    return α


def beta_base_case(λ):
    N = len(λ['B'])
    # return np.ones(N)
    return np.repeat(1/3, 3)  # testing


def beta_inductive_step(λ, O_tplus1, β_tplus1):
    N = len(λ['B'])
    a = λ['A']
    B = λ['B']
    b = [B[state] for state in B.keys()]

    β_t = np.zeros(N)
    for i in range(N):
        β_t[i] = sum([a[i, j] * b[j][O_tplus1] * β_tplus1[j]
                      for j in range(N)])

    return β_t


def beta(O, λ):
    T = len(O)
    N = len(λ['B'])
    β = np.zeros((T, N))
    β[T-1] = beta_base_case(λ)  # -1 because 0 indexed
    for t in range(T-2, -1, -1):  # argh 0 index and non inclusive ranges
        β[t] = beta_inductive_step(λ, O[t+1], β[t+1])

    return β


def delta_base_case(o1, λ):
    π = 1/3  # same initial probability for all states atm
    b = [b for b in λ['B'].values()]
    N = len(λ['B'])

    δ1 = [π*b[i][o1] for i in range(N)]

    return δ1


def psi_base_case():
    ψ1 = np.array([np.nan, np.nan, np.nan])
    return ψ1


def delta_inductive_step(previous_delta, o_t, λ):
    N = len(λ['B'])
    a = λ['A']
    b = [b for b in λ['B'].values()]
    δt = np.zeros(N)
    pδ = previous_delta

    for j in range(N):
        δt[j] = np.max([pδ[i]*a[i,j] for i in range(N)]) * b[j][o_t]

    return δt


def psi_inductive_step(previous_delta, λ):
    N = len(λ['B'])
    a = λ['A']
    ψt = np.zeros(N)

    for j in range(N):
        ψt[j] = np.argmax([previous_delta[i] * a[i,j] for i in range(N)])

    return ψt


def delta_and_psi_and_q(O, λ):
    N = len(λ['B'])
    T = len(O)
    δ = np.zeros((T, N))
    ψ = np.zeros((T, N))

    δ[0] = delta_base_case(O[0], λ)
    ψ[0] = psi_base_case()

    for t in range(1, T):
        δ[t] = delta_inductive_step(δ[t-1], O[t], λ)
        ψ[t] = psi_inductive_step(δ[t-1], λ)

    q = np.zeros(T)
    q[T-1] = np.argmax(δ[T-1])

    for t in range(T-2, -1, -1):  # i hate how confusing this becomes
        q[t] = ψ[t+1][int(q[t+1])]

    return δ, ψ, q


def xi(α, β, λ, O):
    N = len(λ['B'])  # still len(b), is there a better way? this works
    T = len(O)
    a = λ['A']
    b = λ['B']
    b = [b for b in λ['B'].values()]
    ξ = np.zeros((T-1, N, N))  # t, i, j

    from itertools import product
    ξsum = 0
    # t, i, j = (4, 0, 0)
    for t, i, j in product(range(T-1), range(N), range(N)):
        # print(t, i, j)
        ξsum += α[t][i] * a[i, j] * b[j][O[t+1]] * β[t+1][j]

    # probably some smarter way to combine the calculations since we already loop through everything but can do that later.
    # i guess we could save all the results above in ξ[t, i, j] and then divide all values by the final sum? yeah should work.

    for t, i, j in product(range(T-1), range(N), range(N)):
        ξ[t, i, j] = (α[t][i] * a[i, j] * b[j][O[t+1]] * β[t+1][j]) / ξsum

    return ξ


def gamma(O, λ, α, β):
    N = len(λ['B'])
    T = len(O)
    γ = np.zeros((T, N))
    γsum = 0  # the normaliser sum

    for t, j in product(range(T), range(N)):
        γsum += α[t][j] * β[t][j]

    # again we can simplify/effectivise this later

    for t, i in product(range(T), range(N)):
        γ[t, i] = (α[t][i] * β[t][i]) / γsum

    return γ


def lambda_step_update(γ, ξ, O):
    π = γ[0]  # expected frequency in Si at time t = 1 (t=0 with 0-index)
    N = len(γ[0])  # number of states
    T = len(O)

    a = np.zeros((N, N))  # transition matrix
    for i, j in product(range(N), range(N)):
        # sums are T-1 in math, but -2 because 0index
        ξsum = sum([ξ[t, i, j] for t in range(T-2)])
        γsum = sum([γ[t][i] for t in range(T-2)])
        # print(ξsum)
        # print(γsum)
        a[i, j] = ξsum / γsum

    # k = number of unique observation possibilities (types/categories)
    # v = observation vocabulary
    V = ['red', 'green', 'blue']
    K = len(V)
    b = np.zeros((N, K))  # states are rows and k = [red, green, blue]
    for j in range(N):
        for k in range(K):
            v_k = V[k]  # red if k=0, green if k=1, blue if k=2
            # now pick only the γ[t][j] where o[t] was = v_k
            # select from our actual observations sequence $O$, first `red` observations and put those in nominator, sum up and then do same for green and blue
            timesteps = [t for t in range(T) if O[t] == v_k]
            sum_obs = sum(γ[t][j] for t in timesteps)
            sum_all = sum(γ[t][j] for t in range(T))
            b[j][k] = sum_obs / sum_all

    λ_new = {'A': a, 'B': b, 'π': π}

    return λ_new


########################################
# helper/test functions from hmm notes #
########################################

def prob_of_states_given_lambda(Q, λ):
    """

    """
    a = λ['A']
    state_map = {
        'S1': 0,
        'S2': 1,
        'S3': 2
    }
    q = [state_map[state] for state in Q]
    # a_q1q2 -> a[q[0], q[1]]
    T = len(Q)
    probs = [a[q[t], q[t+1]] for t in range(T-1)]
    pi = 1/3  # we assume random starting state
    return pi * np.product(probs)


def prob_of_obs_given_states_and_lambda(O, Q, λ):
    B = λ['B']
    T = len(O)
    probs = [B[Q[t]][O[t]] for t in range(T)]
    return np.product(probs)


def prob_of_obs_given_lambda(O, λ):
    T = len(O)
    state_count = len(λ['B'])
    sum = 0

    for q in product(['S1', 'S2', 'S3'], repeat=T):
        p_q_λ = prob_of_states_given_lambda(q, λ)
        p_o_qλ = prob_of_obs_given_states_and_lambda(O, q, λ)
        sum += p_q_λ * p_o_qλ

    return sum
