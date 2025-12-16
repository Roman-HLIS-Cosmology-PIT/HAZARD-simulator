#FFRNG
import copy
import numpy as np

class FastForwardRNG:
    """
    A deterministic RNG wrapper around NumPy's PCG64 that supports:
    - fast skipping to any step
    - saving/restoring state
    - reproducible sequence indexing
    """

    def __init__(self, seed=12345):
        self.bitgen = np.random.PCG64(seed)
        self.rng = np.random.Generator(self.bitgen)
        self.position = 0  # track how many draws we've advanced

    def random(self):
        """Return a random float in [0,1)."""
        self.position += 1
        return self.rng.random()

    def advance(self, n):
        """
        Skip ahead n draws deterministically.
        Uses PCG64.advance(), which applies the transition polynomial
        in O(log n) time. Supports up to 2^128 skip size.
        """
        self.bitgen.advance(n)
        self.position += n

    def jump_power_of_two(self, k=1):
        """
        Jump ahead by k * 2^128 draws using PCG64.jump().
        Useful for generating independent sequences.
        """
        for _ in range(k):
            self.bitgen.jump()
        self.rng = np.random.Generator(self.bitgen)
        # we don't know exact length of jump, so don't increment position

    def get_state(self):
        """Return a serializable dict containing full RNG state."""
        return {
            "bitgen_state": self.bitgen.state,
            "position": self.position
        }

    def set_state(self, state):
        """Restore RNG state."""
        self.bitgen.state = state["bitgen_state"]
        self.position = state["position"]
        self.rng = np.random.Generator(self.bitgen)

    def goto(self, target_position):
        """
        Fast-forward (or rewind) to an exact index target_position
        in the deterministic random sequence.
        """
        delta = target_position - self.position
        self.advance(delta)

    def spawn_generators_by_jump(self, n_streams: int):
        """
        Create n_streams independent numpy Generators using PCG64.jump().
        Each stream is separated by 2**128 steps (very strong separation).
        """
        gens = []
        # copy the bitgen state so we don't mutate the parent stream
        bitgen = np.random.PCG64()
        bitgen.state = copy.deepcopy(self.bitgen.state)

        for _ in range(n_streams):
            gens.append(np.random.Generator(bitgen))
            bitgen = bitgen.jumped()  # new independent stream
        return gens