from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

def simulate_queue_trajectory(
	lam: float,
	mu: float,
	total_time: float,
	dt: float = 0.01,
	initial_queue: int = 0,
	seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:

	p_arrival = lam * dt
	p_depart = mu * dt

	rng = np.random.default_rng(seed)

	times = np.arange(0, total_time + dt, dt)
	queue = np.zeros(times.shape[0], dtype=int)
	queue[0] = max(0, int(initial_queue))

	for i in range(1, len(times)):
		q_prev = queue[i - 1]

		arrival = rng.random() < p_arrival
		depart = (q_prev > 0) and (rng.random() < p_depart)

		q_new = q_prev + int(arrival) - int(depart)
		queue[i] = max(0, q_new)

	return times, queue


def simulate_many_and_cv(
	lam: float,
	mu: float,
	total_time: float,
	n_samples: int = 500,
	dt: float = 0.01,
	initial_queue: int = 0,
	seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if n_samples <= 1:
		raise ValueError("n_samples must be > 1")

	base_rng = np.random.default_rng(seed)
	child_seeds = base_rng.integers(0, 2**32 - 1, size=n_samples, dtype=np.uint64)

	trajectories = []
	times_ref = None

	for s in child_seeds:
		times, q = simulate_queue_trajectory(
			lam=lam,
			mu=mu,
			total_time=total_time,
			dt=dt,
			initial_queue=initial_queue,
			seed=int(s),
		)
		if times_ref is None:
			times_ref = times
		trajectories.append(q)

	queue_matrix = np.vstack(trajectories)  # shape: (n_samples, n_timepoints)
	mean_q = queue_matrix.mean(axis=0)
	std_q = queue_matrix.std(axis=0, ddof=1)


	cv_q = np.where(mean_q > 0, std_q / mean_q, np.nan)

	return times_ref, mean_q, std_q, cv_q


def plot_cv_vs_time(times: np.ndarray, cv_q: np.ndarray) -> None:
	plt.figure(figsize=(8, 4.5))
	plt.plot(times, cv_q, lw=2)
	plt.xlabel("Time (minutes)")
	plt.ylabel("Coefficient of variation (CV)")
	plt.title("Queue length CV vs time")
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":

	lam = 2.0          # arrivals per minute
	mu = 2.0           # services per minute
	total_time = 60.0  # minutes
	n_samples = 500
	dt = 0.01

	times, mean_q, std_q, cv_q = simulate_many_and_cv(
		lam=lam,
		mu=mu,
		total_time=total_time,
		n_samples=n_samples,
		dt=dt,
		initial_queue=0,
		seed=42,
	)

	print(f"Simulated {n_samples} trajectories up to {total_time} minutes")
	print(f"Final mean queue length: {mean_q[-1]:.3f}")
	print(f"Final std queue length:  {std_q[-1]:.3f}")
	print(f"Final CV:                {cv_q[-1]:.3f}")

	plot_cv_vs_time(times, cv_q)
